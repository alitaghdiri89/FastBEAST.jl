using StaticArrays
using ParallelKMeans
using LinearAlgebra
using ClusterTrees

struct Data{I,D,F,T}
    sector::I
    ct::SVector{D,F}
    hs::F
    values::Vector{T}
end

struct NminTree{D} <: ClusterTrees.PointerBasedTrees.APBTree
    nodes::Vector{ClusterTrees.LevelledTrees.HNode{D}}
    root::Int
    num_elements::Int
    levels::Vector{Int}
end

function NminTree(num_elements; center=SVector(0.0, 0.0, 0.0), radius=0.0)
    root = ClusterTrees.LevelledTrees.HNode(
        ClusterTrees.PointerBasedTrees.Node(Data(0, center, radius, Int[]), 0, 0, 0, 0), 0
    )

    return NminTree([root], 1, num_elements, Int[1])
end

ClusterTrees.root(tree::NminTree{D}) where {D} = tree.root
ClusterTrees.data(tree::NminTree{D}, node) where {D} = tree.nodes[node].node.data
ClusterTrees.parent(tree::NminTree{D}, node) where {D} = tree.nodes[node].node.parent
ClusterTrees.PointerBasedTrees.nextsibling(tree::NminTree, node) =
    tree.nodes[node].node.next_sibling
ClusterTrees.PointerBasedTrees.firstchild(tree::NminTree, node) =
    tree.nodes[node].node.first_child

function value(tree::NminTree{D}, node::Int) where {D}
    if !ClusterTrees.haschildren(tree, node)
        return tree.nodes[node].node.data.values
    else
        values = Int[]
        for leave::Int in ClusterTrees.leaves(tree, node)
            append!(values, tree.nodes[leave].node.data.values)
        end
        return values
    end
end

# BoxTree
struct BoxTreeOptions{I}
    nchildren::I
end

function BoxTreeOptions(; dim=3)
    return BoxTreeOptions(2^dim)
end

function boundingbox(points::Vector{SVector{D,F}}) where {D,F}
    min_dim = Vector(points[1])
    max_dim = Vector(points[1])

    for i in 1:length(points)
        for j in 1:D
            min_dim[j] = min_dim[j] < points[i][j] ? min_dim[j] : points[i][j]
            max_dim[j] = max_dim[j] > points[i][j] ? max_dim[j] : points[i][j]
        end
    end

    center = @MVector zeros(D)

    length_dim = zeros(F, D)
    for j in 1:D
        length_dim[j] = max_dim[j] - min_dim[j]
        center[j] = (max_dim[j] + min_dim[j]) / F(2.0)
    end

    halflength = maximum(length_dim) / F(2.0)

    return halflength, SVector(center)
end

function sector(pt::SVector{3,F}, ct::SVector{3,F}) where {F}
    bl = pt .> ct
    sc = sum(b ? 2^(i - 1) : 0 for (i, b) in enumerate(bl))
    return sc
end

function center(sibling::I, parent_ct::SVector{3,F}, hs::F) where {I,F}
    ct = [
        parent_ct + SVector(-hs, -hs, -hs),
        parent_ct + SVector(hs, -hs, -hs),
        parent_ct + SVector(-hs, hs, -hs),
        parent_ct + SVector(hs, hs, -hs),
        parent_ct + SVector(-hs, -hs, hs),
        parent_ct + SVector(hs, -hs, hs),
        parent_ct + SVector(-hs, hs, hs),
        parent_ct + SVector(hs, hs, hs),
    ]

    return ct[sibling]
end

function clustering(
    points::Vector{SVector{D,F}},
    point_idcs::Vector{I},
    parent_ct::SVector{D,F},
    hs::F,
    treeoptions::BoxTreeOptions{I},
) where {I,D,F}
    #computing true hs
    hs = hs / sqrt(D)
    sorted_point_idcs = zeros(Int, length(point_idcs) + 1, treeoptions.nchildren)
    cluster = [sector(points[idc], parent_ct) for idc in point_idcs]

    for (index, value) in enumerate(cluster)
        sorted_point_idcs[1, value + 1] += 1
        sorted_point_idcs[sorted_point_idcs[1, value + 1] + 1, value + 1] = point_idcs[index]
    end

    cts = [center(i, parent_ct, hs / 2) for i in 1:treeoptions.nchildren]
    #computing "radius" of cube
    hss = sqrt(3) * [hs / 2 for i in 1:treeoptions.nchildren]

    return sorted_point_idcs, cts, hss
end

#KMeansTree
struct KMeansSettings{I}
    max_iters::I
    n_threads::I
end

function KMeansSettings(; max_iters=100, n_threads=1)
    return KMeansSettings(max_iters, n_threads)
end

struct KMeansTreeOptions{I}
    nchildren::I
    settings::KMeansSettings{I}
end

function KMeansTreeOptions(; nchildren=2, settings=KMeansSettings())
    return KMeansTreeOptions(nchildren, settings)
end

function clustering(
    points::Matrix{F},
    point_idcs::Vector{I},
    parent_ct::SVector{3,F},
    hs::F,
    treeoptions::KMeansTreeOptions{I},
) where {I,F}
    kmcluster = ParallelKMeans.kmeans(
        points[:, point_idcs],
        treeoptions.nchildren;
        max_iters=treeoptions.settings.max_iters,
        n_threads=treeoptions.settings.n_threads,
    )

    sorted_point_idcs = zeros(Int, length(point_idcs) + 1, treeoptions.nchildren)

    for (index, value) in enumerate(kmcluster.assignments)
        sorted_point_idcs[1, value] += 1
        sorted_point_idcs[sorted_point_idcs[1, value] + 1, value] = point_idcs[index]
    end

    cts = [SVector{3,F}(kmcluster.centers[:, i]) for i in 1:treeoptions.nchildren]
    hss = [
        maximum(
            norm.(
                eachcol(
                    points[:, sorted_point_idcs[2:(sorted_point_idcs[1, i] + 1), i]] .-
                    kmcluster.centers[:, i],
                ),
            ),
        ) for i in 1:treeoptions.nchildren
    ]

    return sorted_point_idcs, cts, hss
end

function child!(tree::NminTree{D}, treeoptions, state, destination) where {D}
    nmin, maxlevel = destination
    parent_node_idx, ct, hs, level, sibling_idx, point_idcs, points = state
    sorted_point_idcs, cts, hss = clustering(points, point_idcs, ct, hs, treeoptions)

    if sorted_point_idcs[1, 1] == 0
        state_sibling = (
            parent_node_idx, cts, hss, level, sibling_idx + 1, sorted_point_idcs, points
        )
        tree.nodes[parent_node_idx].node.first_child = sibling!(
            tree, treeoptions, state_sibling, destination
        )
    else
        push!(
            tree.nodes,
            ClusterTrees.LevelledTrees.HNode(
                ClusterTrees.PointerBasedTrees.Node(
                    Data(sibling_idx, cts[sibling_idx], hss[sibling_idx], Int[]),
                    treeoptions.nchildren,
                    0,
                    parent_node_idx,
                    0,
                ),
                level,
            ),
        )

        node_idx = length(tree.nodes)
        level >= length(tree.levels) && resize!(tree.levels, level + 1)
        tree.levels[level + 1] = node_idx
        tree.nodes[parent_node_idx].node.first_child = node_idx
        state_sibling = (
            parent_node_idx, cts, hss, level, sibling_idx + 1, sorted_point_idcs, points
        )
        tree.nodes[node_idx].node.next_sibling = sibling!(
            tree, treeoptions, state_sibling, destination
        )

        # Check if more than one node is left
        if sorted_point_idcs[1, 1] == 1 ||
            sorted_point_idcs[1, 1] < nmin ||
            level == maxlevel
            append!(
                tree.nodes[node_idx].node.data.values,
                sorted_point_idcs[(2:(sorted_point_idcs[1, 1] + 1)), 1],
            )

            return node_idx
        else
            state_child = (
                node_idx,
                cts[sibling_idx],
                hss[sibling_idx],
                level + 1,
                1,
                sorted_point_idcs[(2:(sorted_point_idcs[1, 1] + 1)), 1],
                points,
            )
            tree.nodes[node_idx].node.first_child = child!(
                tree, treeoptions, state_child, destination
            )

            return node_idx
        end
    end
end

function sibling!(tree::NminTree{D}, treeoptions, state, destination) where {D}
    nmin, maxlevel = destination
    parent_node_idx, cts, hss, level, sibling_idx, sorted_point_idcs, points = state
    # Enough siblings?
    sibling_idx > treeoptions.nchildren && return 0

    if sorted_point_idcs[1, sibling_idx] == 0
        state_sibling = (
            parent_node_idx, cts, hss, level, sibling_idx + 1, sorted_point_idcs, points
        )

        return sibling!(tree, treeoptions, state_sibling, destination)
    end

    push!(
        tree.nodes,
        ClusterTrees.LevelledTrees.HNode(
            ClusterTrees.PointerBasedTrees.Node(
                Data(sibling_idx, cts[sibling_idx], hss[sibling_idx], Int[]),
                treeoptions.nchildren,
                0,
                parent_node_idx,
                0,
            ),
            level,
        ),
    )

    node_idx = length(tree.nodes)
    level >= length(tree.levels) && resize!(tree.levels, level + 1)
    tree.levels[level + 1] = node_idx

    state_sibling = (
        parent_node_idx, cts, hss, level, sibling_idx + 1, sorted_point_idcs, points
    )
    tree.nodes[node_idx].node.next_sibling = sibling!(
        tree, treeoptions, state_sibling, destination
    )

    # Check if more than one node is left
    if sorted_point_idcs[1, sibling_idx] == 1 ||
        sorted_point_idcs[1, sibling_idx] < nmin ||
        level == maxlevel
        append!(
            tree.nodes[node_idx].node.data.values,
            sorted_point_idcs[(2:(sorted_point_idcs[1, sibling_idx] + 1)), sibling_idx],
        )

        return node_idx
    else
        state_child = (
            node_idx,
            cts[sibling_idx],
            hss[sibling_idx],
            level + 1,
            1,
            sorted_point_idcs[(2:(sorted_point_idcs[1, sibling_idx] + 1)), sibling_idx],
            points,
        )
        tree.nodes[node_idx].node.first_child = child!(
            tree, treeoptions, state_child, destination
        )

        return node_idx
    end
end
