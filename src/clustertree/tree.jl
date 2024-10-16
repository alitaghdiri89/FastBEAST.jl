using ClusterTrees
using StaticArrays

struct BoxTreeOptions{I}
    nmin::I
    maxlevel::I
end

function BoxTreeOptions(; nmin=1, maxlevel=10)

    return BoxTreeOptions(nmin, maxlevel)
end

struct KMeansTreeOptions{I}
    nmin::I
    maxlevel::I
    nchildren::I
    KMeansSettings::NminClusterTrees.NminTrees.KMeansSettings{I}
end

function KMeansTreeOptions(; 
    nmin=1, 
    maxlevel=10,
    nchildren=2,
    KMeansSettings=NminClusterTrees.NminTrees.KMeansSettings()
)

    return KMeansTreeOptions(nmin, maxlevel, nchildren, KMeansSettings)
end


function isfar(
    testnode::ClusterTrees.LevelledTrees.HNode{D}, 
    trialnode::ClusterTrees.LevelledTrees.HNode{D};
    η=2.0
) where D
    test_center = testnode.node.data.ct
    test_radius =  testnode.node.data.hs
    trial_center = trialnode.node.data.ct
    trial_radius = trialnode.node.data.hs
    center_dist = norm(test_center - trial_center)
    dist = center_dist - (test_radius + trial_radius)
    
    if 2 * max(test_radius, trial_radius) < η * (dist) && dist != 0
        return true
    else
        return false
    end
end


function listnearfarinteractions(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    block,
    state,
    nears::Vector{Tuple{Int,Int}},
    fars::Vector{Vector{Tuple{Int,Int}}},
    level::Int;
    η=1.5
) where {T}
    isfar(state[1], state[2], η=η) && (push!(fars[level], block); return nothing)
    !ClusterTrees.haschildren(block_tree, block) && (push!(nears, block); return nothing)
    for chd in ClusterTrees.children(block_tree, block)
        children = ClusterTrees.data(block_tree, chd)
        chd_state = (
            block_tree.test_cluster.nodes[chd[1]],
            block_tree.trial_cluster.nodes[chd[2]]
        )
        listnearfarinteractions(block_tree, chd, chd_state, nears, fars, level + 1, η=η)
    end
end


function computeinteractions(tree::ClusterTrees.BlockTrees.BlockTree{T}; η=1.5) where {T}
    nears = Tuple{Int,Int}[]
    num_levels = length(tree.test_cluster.levels)
    fars = [Tuple{Int,Int}[] for l in 1:num_levels]

    root_state = (tree.test_cluster.nodes[1], tree.trial_cluster.nodes[1])
    root_level = 1

    listnearfarinteractions(
        tree, ClusterTrees.root(tree), root_state, nears, fars, root_level, η=η
    )
    return nears, fars
end

function value(tree::NminClusterTrees.NminTrees.NminTree{D}, node::I) where {I, D}
    return NminClusterTrees.NminTrees.value(tree, node)
end
