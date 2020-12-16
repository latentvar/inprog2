


using Revise # lets you change A2funcs without restarting julia!




include("/Users/hhumayun/Dropbox/Courses/duvenaud_csc412/A2_src.jl")
using Plots
using Statistics: mean
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!

using .A2funcs: skillcontour

using .A2funcs: plot_line_equal_skill!

using StatsFuns

# log prior for each setting of skill
# logsum()
log_prior([0.5 0.3; 0.3 0.7])



function log_prior(zs)
    return sum.(factorized_gaussian_log_density(0, 0, zs))
end


## we want to calcualte separately for each 
# set of skills 




# log of the
function logp_a_beats_b(za, zb)
    return logistic(za - zb)
end


# what is the probablity of observing M (win/lose) guven skills 
# and skills prior 


# likelihood = p(X\params)
# Given an array of games and skills -> what
# is the likelihood

# KxN array - array of skills
# each row represents a player with skills
# zs is an array of skills
# games is an array of Mx2

# p(games|z)
# for each of the skill settings what is Probability
# of the outcome (games)
#= 
for each skill setting:
    calc probablity of games
    sum =#
# test_zs -> K skills for N Players
# test_zs -> (K,N)
# Now index each game to get the prob of outcome for each game

test_zs
test_games
all_games_log_likelihood(test_zs, test_games)

test_zs

all_games_log_likelihood

function all_games_log_likelihood(zs, games)
    
    # zs is player skills (so K x N) -> K skills for each player
    # games is games outcome
    # index zs at games to get an array sorted by the winners prob first
    z = zs[games,:]
    likelihoods = logp_a_beats_b.(z[:,1,:], z[:,2,:])
    sum(likelihoods, dims=1)
end




# joint log = mult sum (p(games|z)*p(z))
# then we have that jl =
function joint_log_density(zs, games)
    all_games_log_likelihood(zs, games) + log_prior(zs)
end

B = 15 # number of elements in batch
N = 4 # Total Number of Players
test_zs = randn(4, 15)
test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2


joint_log_density(test_zs, test_games)



@testset "Test shapes of batches for likelihoods" begin
    B = 15 # number of elements in batch
    N = 4 # Total Number of Players
    test_zs = randn(4, 15)
    test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
    @test size(test_zs) == (N, B)
  # batch of priors
    @test size(log_prior(test_zs)) == (1, B)
  # loglikelihood of p1 beat p2 for first sample in batch
    @test size(logp_a_beats_b(test_zs[1,1], test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
    @test size(logp_a_beats_b.(test_zs[1,:], test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
    @test size(all_games_log_likelihood(test_zs, test_games)) == (1, B)
  # batch loglikelihood under joint of evidence and prior
    @test size(joint_log_density(test_zs, test_games)) == (1, B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]', p1_wins), repeat([2,1]', p2_wins)]...)


function s(x)  sin(x) end

get_trig_val(s,d="none")
function get_trig_val(f;d="none")
    print(d)
    f(20)
end

Plots.PlotlyBackend()


function skillcontour2(f; colour=nothing)
    n = 100
    x = range(-3, stop=3, length=n)
    y = range(-3, stop=3, length=n)
    z_grid = Iterators.product(x, y) # meshgrid for contour
    z_grid = reshape.(collect.(z_grid), :, 1) # add single batch dim
    z = f.(z_grid)
    z = getindex.(z, 1)'
    max_z = maximum(z)
    levels = [.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2] .* max_z
    if colour == nothing
        p1 = contour!(x, y, z, fill=false, levels=levels)
    else
        p1 = contour!(x, y, z, fill=false, c=colour, levels=levels, colorbar=false)
    end
    plot!(p1)
end


# Example for how to use contour plotting code

#


plotlyjs()



gr() # Set the backend to Plotly
N = 2 # Total Number of Players
test_zs = randn(2, 1)



plot(title="Example Gaussian Contour Plot",
    xlabel="Player 1 Skill",
    ylabel="Player 2 Skill"
   )

games_t = two_player_toy_games(10, 10)
games_gaussian(zs) = exp(joint_log_density(zs, games_t))
skillcontour!(games_gaussian)

joint_log_density(test_zs, games_t)

example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.], [0.,0.5], zs))



games_gaussian()


example_gaussian(test_zs)

skillcontour2()


plot_line_equal_skill!()





savefig(joinpath("plots", "example_gaussian.pdf"))





test_zs



zs
test_games = repeat([1,2], 10)
test_games




plot(title="One gme Gaussian Contour Plot",
    xlabel="Player 1 Skill",
    ylabel="Player 2 Skill"
   )

l(zs) =  exp(all_games_log_likelihood(zs, two_player_toy_games(1, 0)))


skillcontour2(zs -> exp(log_prior(zs)))

plot_line_equal_skill!()



# TODO: plot prior contours
log_prior([0.3 0.9])

# TODO: plot likelihood contours
all_games_log_likelihood()

joint_log_density()

# TODO: plot joint contours with player A winning 1 game

# TODO: plot joint contours with player A winning 10 games

# TODO: plot joint contours with player A winning 10 games and player B winning 10 games



factorized_gaussian_log_density(toy_mu, toy_ls,[0 1 ] )


# params are variational parameters mu and sigma 
# we know that z=μ+σ*ϵ

length(toy_mu)
# assuming fully factorized normal 
exp.(toy_ls)

toy_ls .* [2 ; 3]

toy_mu
toy_ls
factorized_gaussian_log_density(toy_mu, toy_ls, [0 ; 0 ])

lp(zs) = joint_log_density(zs, two_player_toy_games(10, 10))

lp()
elbo((toy_mu, toy_ls), lp,20)

length((toy_mu, toy_ls)[1])

num_samples = 10

toy_mu .+ randn(2, num_samples) .* toy_ls

function elbo(params, logp, num_samples)
  # ϵ = randn(length(params))
    N = length(params[1])  
    
  # variational parameters
    ϕ_mu = params[1]
    ϕ_ls = params[2]
    # get B - bactch size`
    B = num_samples
    
    # sample from the current zs to calculate expectation 
    samples = ϕ_mu .+ exp.(ϕ_ls) .* randn(N, B)
    
    logp_estimate = logp(samples)
  # this is a fully factorized gaussian 
    logq_estimate = factorized_gaussian_log_density(ϕ_mu, ϕ_ls, samples)
    # noe this is the E[logp - logq]
    return sum(logp_estimate - logq_estimate) / B # should return scalar (hint: average over batch)
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games=two_player_toy_games(1, 0), num_samples=100)
  # TODO: Write a function that takes parameters for q,
  # evidence as an array of game outcomes,
  # and returns the -elbo estimate with num_samples many samples from q
    logp(zs) = joint_log_density(zs, games)
    return -elbo(params, logp, num_samples)
end


# Toy game
num_players_toy = 2
toy_mu = [1.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)


two_player_toy_games(1,0)



f(x) = 2x^2 + 3
gradient(f, 2)

neg_elbo(params) = neg_toy_elbo(params, games=two_player_toy_games(1, 4), num_samples=10)
grad_params = gradient(neg_elbo, toy_params_init)

grad_params .* 0.002

grad_params[1][1]
grad_params[1][2]


fit_toy_variational_dist(toy_params_init, two_player_toy_games(9, 1), num_itrs=10000, num_q_samples=100, lr=0.001)

function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr=1e-2, num_q_samples=100)
    params_cur = init_params
    for i in 1:num_itrs
        neg_elbo(params) = neg_toy_elbo(params, games=toy_evidence, num_samples=num_samples)
        grad_params = gradient(neg_elbo, params_cur)# TODO: gradients of variational objective with respect to parameters
        
        grad_params_mu = grad_params[1][1]
        grad_params_s = grad_params[1][2]
        params_cur = (params_cur[1] - grad_params_mu * lr, params_cur[2] - grad_params_s * lr)# TODO: update paramters with lr-sized step in descending gradient
        @info neg_toy_elbo(params_cur, games=toy_evidence, num_samples=10)  # TODO: report the current elbbo during training
    # TODO: plot true posterior in red and variational in blue
    # hint: call 'display' on final plot to make it display during training
#        plot();
 #       skillcontour!(z -> exp(joint_log_density(params_cur, z), colour=:red)) # plot likelihood contours for target posterior
  #      plot_line_equal_skill()
   #     display(skillcontour!(exp(z -> factorized_gaussian_log_density(params_cur[1], params_cur[2], z))), colour=:blue)# plot likelihood contours for variational posterior
    end
    return params_cur
end





toy_params_init



ds(z) = joint_log_density(toy_params_init, z)

# TODO: fit q with SVI observing player A winning 1 game
# TODO: save final posterior plots

fit_variational_dist(toy_params_init, )

# TODO: fit q with SVI observing player A winning 10 games
# TODO: save final posterior plots

# TODO: fit q with SVI observing player A winning 10 games and player B winning 10 games
# TODO: save final posterior plots

## Question 4
# Load the Data
using MAT

vars = matread("tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr=1e-2, num_q_samples=10)
    params_cur = init_params
    for i in 1:num_itrs
        neg_elbo(params) = neg_toy_elbo(params, games=toy_evidence, num_samples=num_samples)
        grad_params = gradient(neg_elbo, params_cur)# TODO: gradients of variational objective with respect to parameters
      
        grad_params_mu = grad_params[1][1]
        grad_params_s = grad_params[1][2]
        params_cur = (params_cur[1] - grad_params_mu * lr, params_cur[2] - grad_params_s * lr)# TODO: update paramters with lr-sized step in descending gradient
#  @info # TODO: report objective value with current parameters
    end
    return params_cur
end

# TODO: Initialize variational family
init_mu = # random initialziation
init_log_sigma = # random initialziation
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)


# TODO: 10 players with highest mean skill under variational model
# hint: use sortperm

# TODO: joint posterior over "Roger-Federer" and ""Rafael-Nadal""
# hint: findall function to find the index of these players in player_names
