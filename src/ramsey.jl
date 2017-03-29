######################################################################
# utility functions and setup
######################################################################
if ENV["USER"] == "tamas"
    cd(expanduser("~/research/exact-present/fig/"))
    plotlyjs()
else
    warn("in your own environment, set the path and the backend")
end

using Plots
using Parameters
using NLsolve

"Save plot `plt` with given `basename`, appending appropriate extensions."
function saveplot(plt, basename)
    Plots.pdf(plt, basename * ".pdf")
    PlotlyJS.savefig(plt.o, basename * ".svg")
end

"""
Very simple (normalized) Ramsey model with isoelastic production
function and utility.
"""
@with_kw immutable RamseyModel{T}
    θ::T                        # IES
    α::T                        # capital share
    A::T                        # TFP
    ρ::T                        # discount rate
    δ::T                        # depreciation
end

F(model::RamseyModel, k) = (@unpack α, A, δ = model; A*k^α - δ*k)
F′(model::RamseyModel, k) = (@unpack α, A, δ = model; A*α*k^(α-1) - δ)
F′′(model::RamseyModel, k) = (@unpack α, A = model; A*α*(α-1)*k^(α-2))

"""
Return a function that puts the residuals in its second argument for a
parametrization in `model`, at capital level `k`. For use with `nlsolve`.
"""
function residual_function(model::RamseyModel, k)
    @unpack θ, α, A, ρ, δ = model
    Fk = F(model, k)
    F′k = F′(model, k)
    F′′k = F′′(model, k)
    function residual(c01, r)
        c₀, c₁ = c01
        r[1] = c₁/c₀*(Fk-c₀) - 1/θ*(F′k-ρ)
        r[2] = c₁*(F′k-c₁) - 1/θ*(F′′k*c₀ + (F′k-ρ)*c₁)
    end
end

"Return the steady state capital and consumption for the model."
function steady_state(model::RamseyModel)
    @unpack α, A, ρ, δ = model
    k = ((δ+ρ)/(A*α))^(1/(α-1))
    c = F(model, k)
    k, c
end

"An OK starting point for the nonlinear solver."
function guess_c01(model)
    k, c₀ = steady_state(model)
    c₁ = F′(model, k)
    [c₀, c₁]
end

"""
Solve for `c₀` and `c₁` at `k`, using the initial guess. Return the
result as a vector.
"""
function solve_at_k(model, k, initial_c01)
    r! = residual_function(model, k)
    sol = nlsolve(r!, initial_c01)
    @assert NLsolve.converged(sol)
    sol.zero
end

"""
Poor man's homotopy method. Given a `model` and a sequence of
parameters `ks`, and an initial guess for the solution `x₀` for some
`k₀` (which does not have to be in `ks`), solve using `solve_at_k`
with the nearest solution as the initial guess.
"""
function continuation_solver(model, ks, k₀, x₀; norm_p = 2)
    xs = Array{eltype(x₀)}(length(x₀), length(ks))
    i = findmin(norm(k-k₀, norm_p) for k in ks)[2]
    xs[:, i] = solve_at_k(model, ks[i], x₀)
    for i in (i-1):-1:1
        xs[:, i] = solve_at_k(model, ks[i], xs[:, i+1])
    end
    for i in (i+1):length(ks)
        xs[:, i] = solve_at_k(model, ks[i], xs[:, i-1])
    end
    xs
end

"""
Plot tangent with level and slope `c01` at `k`. `ℓ` is the half-length
of tangent line, `s` compensates for scale differences between axes.
"""
function plot_tangent!(plt, k, c01, ℓ, s=1)
    c₀, c₁ = c01
    scatter!(plt, [k], [c₀], markersize = 2)
    h = ℓ/√(s^2+c₁^2)
    hs = [-h, h]
    plot!(plt, k + hs, c₀ + c₁ * hs, seriestype = :line)
end

"""
Second-order approximation for derivative through 3 points.

Approximate `(x₁,y₁)`, `(x₂,y₂)`, `(0,0)` with `y = a x² + b x`, and
return the derivative `b` at `0`.
"""
deriv3point(x₁, y₁, x₂, y₂) = (y₁/x₁*x₂ - y₂/x₂*x₁)/(x₂-x₁)

"""
Approximate `(x,y)` for the given 3 points with `y = a x² + b x + c`,
and at `x[2]`.
"""
function deriv3point(xs, ys)
    deriv3point(xs[1]-xs[2], ys[1]-ys[2],
                xs[3]-xs[2], ys[3]-ys[2])
end

"""
Return vectors of the

- interior `k`s,
- interior approximated `c′(k)`s,
- interior Euler residuals

given the `model`, `k`s, and `c`s.

Uses a quadratic approximation with nearby points.
"""
function residual_quadapprox(model, ks, cs)
    @unpack θ, ρ = model
    r = zeros(length(ks)-2)
    c′ = zeros(length(ks)-2)
    for i in 1:(length(r))
        ixs = i:(i+2)
        k = ks[i+1]
        c = cs[i+1]
        c′[i] = deriv3point(ks[ixs], cs[ixs])
        r[i] = c′[i]/c*(F(model, k)-c) - 1/θ*(F′(model, k)-ρ)
    end
    ks[2:end], c′, r
end

######################################################################
# runtime code
######################################################################

# define a model
model = RamseyModel(θ = 2.0, α = 0.3, A = 1.0, ρ = 0.02, δ = 0.05)

# the steady state
kₛ, cₛ = steady_state(model)
c01ₛ = solve_at_k(model, kₛ, guess_c01(model))

# solve at various k values
ks = linspace(0.01*kₛ, 2*kₛ, 100)
c01s = continuation_solver(model, ks, kₛ, c01ₛ)

# approximation of c′ for nearby points
inner_ks, inner_c′_approx, inner_r = residual_quadapprox(model, ks, c01s[1,:])

# plot of policy function with steady state
plt = plot(ks, c01s[1, :], xlab = "k", ylab = "c(k)", legend = false)
scatter!(plt, [kₛ], [cₛ])
saveplot(plt, "ck")

# plot of tangents c₁
plt = plot(ks, c01s[1, :], xlab = "k", ylab = "c(k)", legend = false)
for i in 10:20:length(ks)
    display(plot_tangent!(plt, ks[i], c01s[:, i], 1, 0.5))
end
saveplot(plt, "ck-tangents")

# plot of c1 and c′
plt = plot(inner_ks, inner_c′_approx, label = "c′ from c₀", xlab="k", ylab = "c′")
plot!(plt, inner_ks, c01s[2, 2:end], label = "c₁")
saveplot(plt, "cderiv")

plt = plot(inner_ks, inner_r, xlab="k", ylab="Euler residual", legend = false)
saveplot(plt, "eulerresid")
