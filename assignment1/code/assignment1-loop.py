import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest
from heapq import heappush, heappop


def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision

    Overall process:
    Reference: https://github.com/mikedh/trimesh/blob/main/trimesh/remesh.py#L207
    1. Calculate odd vertices.
      Assign a new odd vertex on each edge and
      calculate the value for the boundary case and the interior case.
      The value is calculated as follows.
          v2
        / f0 \\        0
      v0--e--v1      /   \\
        \\f1 /     v0--e--v1
          v3
      - interior case : 3:1 ratio of mean(v0,v1) and mean(v2,v3)
      - boundary case : mean(v0,v1)
    2. Calculate even vertices.
      The new even vertices are calculated with the existing
      vertices and their adjacent vertices.
        1---2
       / \\/ \\      0---1
      0---v---3     / \\/ \\
       \\ /\\/    b0---v---b1
        k...4
      - interior case : (1-kB):B ratio of v and k adjacencies
      - boundary case : 3:1 ratio of v and mean(b0,b1)
    3. Compose new faces with new vertices.

    # The following implementation considers only the interior cases
    # You should also consider the boundary cases and more iterations in your submission
    """

    # prepare geometry for the loop subdivision
    vertices, faces = mesh.vertices, mesh.faces  # [N_vertices, 3] [N_faces, 3]

    def _subdivide(vertices, faces):
        # find the unique edges of our faces
        edges, edges_face = faces_to_edges(faces, return_index=True)
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)

        # split edges to interior edges and boundary edges
        edge_inter = np.sort(grouping.group_rows(edges, require_count=2), axis=1)
        edge_bound = grouping.group_rows(edges, require_count=1)

        # set also the mask for interior edges and boundary edges
        edge_bound_mask = np.zeros(len(edges), dtype=bool)
        edge_bound_mask[edge_bound] = True
        edge_bound_mask = edge_bound_mask[unique]
        edge_inter_mask = ~edge_bound_mask

        ###########
        # Step 1: #
        ###########
        # set odd vertices to the middle of each edge (default as boundary case).
        odd = vertices[edges[unique]].mean(axis=1)
        # connect the odd vertices with even vertices
        # however, the odd vertices need further updates over its position
        # we therefore complete this step later afterward.

        ###########
        # Step 2: #
        ###########
        # find v0, v1, v2, v3 and each odd vertex
        # v0 and v1 are at the end of the edge where the generated odd vertex on
        # locate the edge first
        e = edges[unique[edge_inter_mask]]
        # locate the endpoints for each edge
        e_v0 = vertices[e][:, 0]
        e_v1 = vertices[e][:, 1]

        # v2 and v3 are at the farmost position of the two triangle
        # locate the two triangle face
        edge_pair = np.zeros(len(edges)).astype(int)
        edge_pair[edge_inter[:, 0]] = edge_inter[:, 1]
        edge_pair[edge_inter[:, 1]] = edge_inter[:, 0]
        opposite_face1 = edges_face[unique]
        opposite_face2 = edges_face[edge_pair[unique]]
        # locate the corresponding edge
        e_f0 = faces[opposite_face1[edge_inter_mask]]
        e_f1 = faces[opposite_face2[edge_inter_mask]]
        # locate the vertex index and vertex location
        e_v2_idx = e_f0[~(e_f0[:, :, None] == e[:, None, :]).any(-1)]
        e_v3_idx = e_f1[~(e_f1[:, :, None] == e[:, None, :]).any(-1)]
        e_v2 = vertices[e_v2_idx]
        e_v3 = vertices[e_v3_idx]

        # update the odd vertices based the v0, v1, v2, v3, based the following:
        # 3 / 8 * (e_v0 + e_v1) + 1 / 8 * (e_v2 + e_v3)
        odd[edge_inter_mask] = 0.375 * e_v0 + 0.375 * e_v1 + e_v2 / 8.0 + e_v3 / 8.0

        ###########
        # Step 3: #
        ###########
        # find vertex neighbors for even vertices and update accordingly
        neighbors = graph.neighbors(edges=edges[unique], max_index=len(vertices))
        # convert list type of array into a fixed-shaped numpy array (set -1 to empties)
        neighbors = np.array(list(zip_longest(*neighbors, fillvalue=-1))).T
        # if the neighbor has -1 index, its point is (0, 0, 0), so that
        # it is not included in the summation of neighbors when calculating the even
        vertices_ = np.vstack([vertices, [0.0, 0.0, 0.0]])
        # number of neighbors
        k = (neighbors + 1).astype(bool).sum(axis=1)

        # calculate even vertices for the interior case
        # beta = 1 / k * (5 / 8 - (3 / 8 + 1 / 4 * np.cos(2 * np.pi / k)) ** 2)
        beta = (40.0 - (2.0 * np.cos(2 * np.pi / k) + 3) ** 2) / (64 * k)
        even = (
                beta[:, None] * vertices_[neighbors].sum(1)
                + (1 - k[:, None] * beta[:, None]) * vertices
        )

        # calculate even vertices for the boundary case
        if edge_bound_mask.any():
            # boundary vertices from boundary edges
            vrt_bound_mask = np.zeros(len(vertices), dtype=bool)
            vrt_bound_mask[np.unique(edges[unique][~edge_inter_mask])] = True
            # one boundary vertex has two neighbor boundary vertices (set others as -1)
            boundary_neighbors = neighbors[vrt_bound_mask]
            boundary_neighbors[~vrt_bound_mask[neighbors[vrt_bound_mask]]] = -1

            even[vrt_bound_mask] = (
                    vertices_[boundary_neighbors].sum(axis=1) / 8.0
                    + (3.0 / 4.0) * vertices[vrt_bound_mask]
            )

        ############
        # Step 1+: #
        ############
        # complete the subdivision by updating the vertex list and face list

        # the new faces with odd vertices
        odd_idx = inverse.reshape((-1, 3)) + len(vertices)
        new_faces = np.column_stack(
            [
                faces[:, 0],
                odd_idx[:, 0],
                odd_idx[:, 2],
                odd_idx[:, 0],
                faces[:, 1],
                odd_idx[:, 1],
                odd_idx[:, 2],
                odd_idx[:, 1],
                faces[:, 2],
                odd_idx[:, 0],
                odd_idx[:, 1],
                odd_idx[:, 2],
            ]
        ).reshape((-1, 3))

        # stack the new even vertices and odd vertices
        new_vertices = np.vstack((even, odd))

        return new_vertices, new_faces

    for _ in range(iterations):
        vertices, faces = _subdivide(vertices, faces)

    return trimesh.Trimesh(vertices, faces)


if __name__ == '__main__':
    # Load mesh and print information
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    print(f'Mesh Info: {mesh}')

    mesh_subdivided = subdivision_loop(mesh, iterations=1)

    # print the new mesh information and save the mesh
    print(f'Subdivided Mesh Info: {mesh_subdivided}')
    mesh_subdivided.export('assets/cube_subdivided_1.obj')