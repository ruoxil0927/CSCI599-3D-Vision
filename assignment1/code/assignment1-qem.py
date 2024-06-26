import time
import trimesh
from trimesh import graph, grouping
from trimesh.geometry import faces_to_edges
import numpy as np
from itertools import zip_longest
from heapq import heappush, heappop
from collections import defaultdict


class QuadricErrorMetrics:
    def __init__(self, mesh, face_count):
        self.mesh = mesh
        self.face_count = face_count
        self.init_mesh()

    def init_mesh(self):
        # not familiar with TrackedArray, make every thing to numpy
        # Vertices and faces
        vertices, faces = self.mesh.vertices, self.mesh.faces
        self.vertices, self.faces = np.array(vertices), np.array(faces)

        # Edges
        edges, edges_face = faces_to_edges(faces, return_index=True)
        edges.sort(axis=1)
        unique, inverse = grouping.unique_rows(edges)
        self.edges = np.array(edges[unique])

        # Calculate the K for each face
        # 1. Calculate the intercept for each surface
        face_normal = self.mesh.face_normals
        points = self.vertices[self.faces[:, 0]]
        d = -np.sum(face_normal * points, axis=1)
        # 2. v = [a, b, c, d]^T, K = vv^T
        v = np.column_stack((face_normal, d))
        K = np.array([np.outer(v[i], v[i]) for i in range(len(v))])  # TODO: This line can be optimized

        # Calculate the K_i for each vertex
        Ki = np.zeros((len(self.vertices), 4, 4))
        for i in range(len(self.faces)):
            for vertex in faces[i]:
                Ki[vertex] += K[i]
        self.Ki = Ki

        # vertex to face
        self.v2f = defaultdict(list)
        for fi, face in enumerate(self.faces):
            for v in face:
                self.v2f[v].append(fi)
        # vertex to neighbor vertices
        self.v2v = defaultdict(list)
        for u, v in self.edges:
            self.v2v[u].append(v)
            self.v2v[v].append(u)

        # Record modification for vertex, edge and face
        self.deleted_vertices = [False] * len(self.vertices)
        self.deleted_faces = [False] * len(self.faces)

    def compute_cost(self, v1, v2):
        # Follow the solution on the slides
        K1 = self.Ki[v1]
        K2 = self.Ki[v2]
        K = K1 + K2

        B = K[:3, :3]
        w = K[3, :3][:, np.newaxis]  # make w as a column vector
        d2 = K[-1, -1]

        try:
            x = -np.linalg.inv(B) @ w
            x = x.squeeze()
            cost = float(x.T @ B @ x + 2 * w.T @ x + d2)
        except np.linalg.LinAlgError as err:
            vertex_1 = self.vertices[v1][:, None]
            vertex_2 = self.vertices[v2][:, None]
            vertex_mid = (vertex_1 + vertex_2) / 2
            cost_1 = float(vertex_1.T @ B @ vertex_1 + 2 * w.T @ vertex_1 + d2)
            cost_2 = float(vertex_2.T @ B @ vertex_2 + 2 * w.T @ vertex_2 + d2)
            cost_mid = float(vertex_mid.T @ B @ vertex_mid + 2 * w.T @ vertex_mid + d2)

            min_cost = min(cost_1, cost_2, cost_mid)
            if cost_1 == min_cost:
                cost, x = cost_1, vertex_1
            elif cost_2 == min_cost:
                cost, x = cost_2, vertex_2
            else:
                cost, x = cost_mid, vertex_mid

        return cost, x

    def solve(self):
        edge_heap = []
        # Initialize the min heap
        for v1, v2 in self.edges:
            cost, _ = self.compute_cost(v1, v2)
            heappush(edge_heap, (cost, v1, v2))

        # Update the mesh until meet the requirement
        while edge_heap and len(self.deleted_faces) - np.sum(self.deleted_faces) > self.face_count:
            cost, v1, v2 = heappop(edge_heap)

            curr_cost, vbar = self.compute_cost(v1, v2)
            # If either vertex has been deleted
            if self.deleted_vertices[v1] or self.deleted_vertices[v2] or cost != curr_cost:
                continue

            # Update the two vertices
            # Set v1 = vbar and delete v2
            self.vertices[v1] = self.vertices[v2] = vbar
            self.deleted_vertices[v2] = True
            # Update the Ki[v1] = Ki[v1] + Ki[v2]
            self.Ki[v1] += self.Ki[v2]

            # Update all faces
            for fi in self.v2f[v2]:
                if v1 in self.faces[fi]:
                    # both v1 and v2 in the face, this face should be deleted
                    self.deleted_faces[fi] = True
                else:
                    for i in range(3):
                        # update all v2 to v1
                        if self.faces[fi][i] == v2:
                            self.faces[fi][i] = v1

            # Update all edges
            # update v2 to v1
            for vi in self.v2v[v2]:
                if vi == v1:
                    continue
                if vi not in self.v2v[v1]:
                    self.v2v[v1].append(vi)
                self.v2v[vi].remove(v2)
                if v1 not in self.v2v[vi]:
                    self.v2v[vi].append(v1)

            # recalculate the cost and add to the min heap
            for vi in self.v2v[v1]:
                vs, vl = min(vi, v1), max(vi, v1)
                cost, _ = self.compute_cost(vs, vl)
                heappush(edge_heap, (cost, vs, vl))

        # Rebuild the mesh
        # we already set the merged vertices to the same coordinate
        new_faces = []
        for i, f in enumerate(self.faces):
            if not self.deleted_faces[i]:
                new_faces.append(f)
        new_faces = np.array(new_faces)

        new_mesh = trimesh.Trimesh(vertices=self.vertices, faces=new_faces)
        return new_mesh


if __name__ == '__main__':
    mesh = trimesh.creation.box(extents=[2, 2, 2])

    face_count = 6

    mesh_decimated = mesh.simplify_quadric_decimation(face_count)
    mesh_decimated.export('assets/cube_decimated.obj')

    sol = QuadricErrorMetrics(mesh, face_count)
    start = time.time()
    mesh_decimated_custom = sol.solve()
    end = time.time()
    print(f'Running time: {end - start} s')
    mesh_decimated_custom.export('assets/cube_decimated_custom.obj')
