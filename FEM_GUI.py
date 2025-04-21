import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Streamlit page config ---
st.set_page_config(layout="wide")
st.title("2D Heterogeneous Plate FEM Structural Analysis")

# --- 1. Geometry and Mesh Parameters ---
st.header("1. Geometry and Mesh Parameters")
Lx = st.number_input("Plate Length (Lx, m)", value=8.0)
Ly = st.number_input("Plate Height (Ly, m)", value=4.0)
nx_elem = st.number_input("Number of elements along x", value=48, min_value=1)
ny_elem = st.number_input("Number of elements along y", value=24, min_value=1)
elem_type = st.selectbox("Element type", ["quad", "tri"])
analysis_type = st.selectbox("Analysis type", ["plane_stress", "plane_strain"])

# --- 2. Inclusion Geometry ---
st.header("2. Inclusion Geometry")
incl_x_min = st.number_input("Inclusion x min", value=3.0)
incl_x_max = st.number_input("Inclusion x max", value=5.0)
incl_y_min = st.number_input("Inclusion y min", value=1.5)
incl_y_max = st.number_input("Inclusion y max", value=2.5)

# --- 3. Material Properties ---
st.header("3. Material Property Functions")
E_func_code = st.text_area(
    "Young's modulus function E_func(x, y):",
    value=(
        "if (incl_x_min <= x <= incl_x_max) and (incl_y_min <= y <= incl_y_max):\n"
        "    return 300e9\n"
        "else:\n"
        "    return 150e9*np.sin(np.pi*x/Lx)"
    ),
    height=100,
)
nu_func_code = st.text_area(
    "Poisson's ratio function nu_func(x, y):",
    value=(
        "if (incl_x_min <= x <= incl_x_max) and (incl_y_min <= y <= incl_y_max):\n"
        "    return 0.25\n"
        "else:\n"
        "    return 0.35"
    ),
    height=100,
)

# --- 4. Boundary Conditions ---
st.header("4. Boundary Conditions")
st.markdown("#### Dirichlet and Point Load BCs (bc_list)")
dirichlet_code = st.text_area(
    "List of Dirichlet/Point BC dictionaries (Python list):",
    value=(
        "[\n"
        "    {'line_func': lambda x, y: x-0, 'range': {'y': (0, Ly)}, 'type': 'Dirichlet', 'value': lambda x, y: [0.0, 0.0]},\n"
        "    {'line_func': lambda x, y: x-Lx, 'range': {'y': (0, Ly)}, 'type': 'Dirichlet', 'value': lambda x, y: [0.0, 0.0]},\n"
        "]"
    ),
    height=100,
)

st.markdown("#### Traction BCs (traction_list, distributed loads)")
neumann_code = st.text_area(
    "List of Traction BC dictionaries (Python list):",
    value=(
        "[\n"
        "    {'line_func': lambda x, y: x-Lx/2, 'range': {'y': (0, Ly)}, 'value': lambda x, y: np.array([0.625e6, 0.0])},\n"
        "]"
    ),
    height=100,
)

# --- 5. Deformation scale input ---
st.header("5. Deformation Scale Factor")
scale = st.number_input("Deformation scale factor (e.g. 1000)", value=1000.0)

# --- Run Button ---
run = st.button("Run FEM Analysis")

if run:
    st.header("FEM Solution")

    # --- Compile user material functions ---
    local_vars = dict(
        Lx=Lx,
        Ly=Ly,
        incl_x_min=incl_x_min,
        incl_x_max=incl_x_max,
        incl_y_min=incl_y_min,
        incl_y_max=incl_y_max,
        np=np,
        pi=np.pi,
    )
    exec(
        f"def E_func(x, y):\n    " + "\n    ".join(E_func_code.splitlines()), local_vars
    )
    exec(
        f"def nu_func(x, y):\n    " + "\n    ".join(nu_func_code.splitlines()),
        local_vars,
    )
    E_func = local_vars["E_func"]
    nu_func = local_vars["nu_func"]

    # --- Compile BC lists ---
    try:
        bc_list = eval(dirichlet_code, local_vars)
        traction_list = eval(neumann_code, local_vars)
    except Exception as e:
        st.error(f"Error in BC definitions: {e}")
        st.stop()

    # --- FEM Implementation (EXACT LOGIC FROM YOUR NOTEBOOK) ---
    tol = 1e-8

    # Mesh Generation
    if elem_type == "quad":
        dx = Lx / nx_elem
        dy = Ly / ny_elem
        nx_nodes = nx_elem + 1
        ny_nodes = ny_elem + 1
        x_coords = np.linspace(0, Lx, nx_nodes)
        y_coords = np.linspace(0, Ly, ny_nodes)
        X, Y = np.meshgrid(x_coords, y_coords)
        coords = np.column_stack([X.flatten(), Y.flatten()])
        n_elem = nx_elem * ny_elem
        conn = np.zeros((n_elem, 4), dtype=int)
        elem = 0
        for j in range(ny_elem):
            for i in range(nx_elem):
                n1 = j * nx_nodes + i
                n2 = n1 + 1
                n3 = n2 + nx_nodes
                n4 = n1 + nx_nodes
                conn[elem, :] = [n1, n2, n3, n4]
                elem += 1
    elif elem_type == "tri":
        nx_nodes = nx_elem + 1
        ny_nodes = ny_elem + 1
        x_coords = np.linspace(0, Lx, nx_nodes)
        y_coords = np.linspace(0, Ly, ny_nodes)
        X, Y = np.meshgrid(x_coords, y_coords)
        coords = np.column_stack([X.flatten(), Y.flatten()])
        conn = []
        for j in range(ny_elem):
            for i in range(nx_elem):
                n1 = j * nx_nodes + i
                n2 = n1 + 1
                n3 = n2 + nx_nodes
                n4 = n1 + nx_nodes
                conn.append([n1, n2, n3])
                conn.append([n1, n3, n4])
        conn = np.array(conn, dtype=int)
        n_elem = conn.shape[0]
    else:
        st.error("Unknown element type")
        st.stop()
    n_nodes = coords.shape[0]

    # --- FEM routines (from your notebook, unchanged) ---
    def constitutive_matrix(E, nu, analysis_type):
        if analysis_type == "plane_stress":
            coeff = E / (1 - nu**2)
            D = coeff * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        elif analysis_type == "plane_strain":
            coeff = E / ((1 + nu) * (1 - 2 * nu))
            D = coeff * np.array(
                [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]]
            )
        else:
            raise ValueError("Unknown analysis type.")
        return D

    def shape_quad(xi, eta):
        N = np.array(
            [
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            ]
        )
        dN_dxi = np.array(
            [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)]
        )
        dN_deta = np.array(
            [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
        )
        return N, dN_dxi, dN_deta

    def elemental_stiffness_quad(node_coords, analysis_type):
        Ke = np.zeros((8, 8))
        gp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        w = np.array([1.0, 1.0])
        for i in range(2):
            for j in range(2):
                xi = gp[i]
                eta = gp[j]
                weight = w[i] * w[j]
                N, dN_dxi, dN_deta = shape_quad(xi, eta)
                J = np.zeros((2, 2))
                for a in range(4):
                    J[0, 0] += dN_dxi[a] * node_coords[a, 0]
                    J[0, 1] += dN_dxi[a] * node_coords[a, 1]
                    J[1, 0] += dN_deta[a] * node_coords[a, 0]
                    J[1, 1] += dN_deta[a] * node_coords[a, 1]
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)
                dN_dx = np.zeros(4)
                dN_dy = np.zeros(4)
                for a in range(4):
                    dN = np.array([dN_dxi[a], dN_deta[a]])
                    dN_xy = invJ @ dN
                    dN_dx[a] = dN_xy[0]
                    dN_dy[a] = dN_xy[1]
                B = np.zeros((3, 8))
                for a in range(4):
                    B[0, 2 * a] = dN_dx[a]
                    B[1, 2 * a + 1] = dN_dy[a]
                    B[2, 2 * a] = dN_dy[a]
                    B[2, 2 * a + 1] = dN_dx[a]
                x_gp = sum(N[a] * node_coords[a, 0] for a in range(4))
                y_gp = sum(N[a] * node_coords[a, 1] for a in range(4))
                E_val = E_func(x_gp, y_gp)
                nu_val = nu_func(x_gp, y_gp)
                D = constitutive_matrix(E_val, nu_val, analysis_type)
                Ke += B.T @ D @ B * detJ * weight
        return Ke

    def shape_tri(x, y, node_coords):
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]
        A = 0.5 * np.linalg.det(np.array([[1, x1, y1], [1, x2, y2], [1, x3, y3]]))
        beta = np.array([y2 - y3, y3 - y1, y1 - y2])
        gamma = np.array([x3 - x2, x1 - x3, x2 - x1])
        dN_dx = beta / (2 * A)
        dN_dy = gamma / (2 * A)
        B = np.zeros((3, 6))
        for a in range(3):
            B[0, 2 * a] = dN_dx[a]
            B[1, 2 * a + 1] = dN_dy[a]
            B[2, 2 * a] = dN_dy[a]
            B[2, 2 * a + 1] = dN_dx[a]
        return A, B

    def elemental_stiffness_tri(node_coords, analysis_type):
        Ke = np.zeros((6, 6))
        A, B = shape_tri(0, 0, node_coords)
        x_c = np.mean(node_coords[:, 0])
        y_c = np.mean(node_coords[:, 1])
        E_val = E_func(x_c, y_c)
        nu_val = nu_func(x_c, y_c)
        D = constitutive_matrix(E_val, nu_val, analysis_type)
        Ke = B.T @ D @ B * A
        return Ke

    def elemental_traction(edge_coords, q_func):
        gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        w = [1, 1]
        Te = np.zeros(4)
        length = np.linalg.norm(edge_coords[1] - edge_coords[0])
        for i in range(2):
            xi = gp[i]
            weight = w[i]
            N_edge = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
            x = N_edge[0] * edge_coords[0, 0] + N_edge[1] * edge_coords[1, 0]
            y = N_edge[0] * edge_coords[0, 1] + N_edge[1] * edge_coords[1, 1]
            qx, qy = q_func(x, y)
            dS = (length / 2) * weight
            Te[0] += N_edge[0] * qx * dS
            Te[1] += N_edge[0] * qy * dS
            Te[2] += N_edge[1] * qx * dS
            Te[3] += N_edge[1] * qy * dS
        return Te

    # --- Global Assembly (from your notebook) ---
    ndof = 2 * n_nodes
    Kg_global = np.zeros((ndof, ndof))
    Fg_global = np.zeros(ndof)
    Tg_global = np.zeros(ndof)

    for e in range(n_elem):
        nodes_e = conn[e]
        node_coords = coords[nodes_e]
        # 1) stiffness
        if elem_type == "quad":
            Ke = elemental_stiffness_quad(node_coords, analysis_type)
            dofs = np.vstack([2 * nodes_e, 2 * nodes_e + 1]).T.flatten()
            for i_loc, I in enumerate(dofs):
                for j_loc, J in enumerate(dofs):
                    Kg_global[I, J] += Ke[i_loc, j_loc]
            # 2) traction on edges
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for a, b in edges:
                n1, n2 = nodes_e[a], nodes_e[b]
                xm, ym = coords[[n1, n2]].mean(axis=0)
                for tr in traction_list:
                    if abs(tr["line_func"](xm, ym)) < tol:
                        ok = True
                        for dim, rng in tr.get("range", {}).items():
                            if dim == "x" and not (rng[0] - tol <= xm <= rng[1] + tol):
                                ok = False
                            if dim == "y" and not (rng[0] - tol <= ym <= rng[1] + tol):
                                ok = False
                        if not ok:
                            continue
                        edge_coords = coords[[n1, n2]]
                        Te = elemental_traction(edge_coords, tr["value"])
                        local_dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
                        for ii, ld in enumerate(local_dofs):
                            Tg_global[ld] += Te[ii]
        elif elem_type == "tri":
            Ke = elemental_stiffness_tri(node_coords, analysis_type)
            dofs = np.vstack([2 * nodes_e, 2 * nodes_e + 1]).T.flatten()
            for i_loc, I in enumerate(dofs):
                for j_loc, J in enumerate(dofs):
                    Kg_global[I, J] += Ke[i_loc, j_loc]
            tri_edges = [(0, 1), (1, 2), (2, 0)]
            for a, b in tri_edges:
                n1, n2 = nodes_e[a], nodes_e[b]
                xm, ym = coords[[n1, n2]].mean(axis=0)
                for tr in traction_list:
                    if abs(tr["line_func"](xm, ym)) < tol:
                        ok = True
                        for dim, rng in tr.get("range", {}).items():
                            if dim == "x" and not (rng[0] - tol <= xm <= rng[1] + tol):
                                ok = False
                            if dim == "y" and not (rng[0] - tol <= ym <= rng[1] + tol):
                                ok = False
                        if not ok:
                            continue
                        edge_coords = coords[[n1, n2]]
                        Te = elemental_traction(edge_coords, tr["value"])
                        local_dofs = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1]
                        for ii, ld in enumerate(local_dofs):
                            Tg_global[ld] += Te[ii]
        else:
            raise ValueError("Element type not recognized.")

    # --- Apply Boundary Conditions (from your notebook) ---
    prescribed_dofs = {}
    for i in range(n_nodes):
        x, y = coords[i]
        for bc in bc_list:
            meets_line = True
            if "line_func" in bc and callable(bc["line_func"]):
                if abs(bc["line_func"](x, y)) > tol:
                    meets_line = False
            meets_range = True
            if "range" in bc:
                rng = bc["range"]
                if "x" in rng:
                    xmin, xmax = rng["x"]
                    if not (xmin - tol <= x <= xmax + tol):
                        meets_range = False
                if "y" in rng:
                    ymin, ymax = rng["y"]
                    if not (ymin - tol <= y <= ymax + tol):
                        meets_range = False
            if meets_line and meets_range:
                bc_value = bc["value"](x, y) if callable(bc["value"]) else bc["value"]
                if bc["type"] == "Dirichlet":
                    prescribed_dofs[2 * i] = bc_value[0]
                    prescribed_dofs[2 * i + 1] = bc_value[1]
                elif bc["type"] == "Neumann":
                    Fg_global[2 * i] += bc_value[0]
                    Fg_global[2 * i + 1] += bc_value[1]
    Ng_global = Fg_global + Tg_global

    # Store copies before BC
    K0 = Kg_global.copy()
    N0 = Ng_global.copy()

    # Impose Dirichlet conditions via elimination
    for dof, value in prescribed_dofs.items():
        Ng_global -= Kg_global[:, dof] * value
        Kg_global[dof, :] = 0.0
        Kg_global[:, dof] = 0.0
        Kg_global[dof, dof] = 1.0
        Ng_global[dof] = value

    # --- Solve System ---
    U = np.linalg.solve(Kg_global, Ng_global)

    # --- Postprocessing: Stresses (EXACT LOGIC FROM YOUR NOTEBOOK) ---
    nodal_stress = np.zeros((n_nodes, 3))  # columns: [sigma_x, sigma_y, tau_xy]
    count = np.zeros(n_nodes)

    if elem_type == "quad":
        # For each quadrilateral element, evaluate stress at each of its 4 nodes.
        # Natural coordinates at nodes: (-1,-1), (1,-1), (1,1), (-1,1)
        natural_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for e in range(n_elem):
            node_ids = conn[e, :]
            node_coords = coords[node_ids, :]  # 4x2 array
            # Assemble the element displacement vector (8 dofs)
            dof_ids = []
            for nid in node_ids:
                dof_ids.extend([2 * nid, 2 * nid + 1])
            u_e = U[dof_ids]
            # Loop over the 4 nodes of the element
            for local_idx, (xi, eta) in enumerate(natural_coords):
                # Evaluate shape functions and their derivatives at (xi, eta)
                N, dN_dxi, dN_deta = shape_quad(xi, eta)
                # Compute the Jacobian matrix
                J = np.zeros((2, 2))
                for a in range(4):
                    J[0, 0] += dN_dxi[a] * node_coords[a, 0]
                    J[0, 1] += dN_dxi[a] * node_coords[a, 1]
                    J[1, 0] += dN_deta[a] * node_coords[a, 0]
                    J[1, 1] += dN_deta[a] * node_coords[a, 1]
                invJ = np.linalg.inv(J)
                # Compute derivatives with respect to x and y
                dN_dx = np.zeros(4)
                dN_dy = np.zeros(4)
                for a in range(4):
                    dN = np.array([dN_dxi[a], dN_deta[a]])
                    dN_xy = invJ @ dN
                    dN_dx[a] = dN_xy[0]
                    dN_dy[a] = dN_xy[1]
                # Build the strain-displacement (B) matrix (size 3x8)
                B = np.zeros((3, 8))
                for a in range(4):
                    B[0, 2 * a] = dN_dx[a]
                    B[1, 2 * a + 1] = dN_dy[a]
                    B[2, 2 * a] = dN_dy[a]
                    B[2, 2 * a + 1] = dN_dx[a]
                # Evaluate physical coordinates at this node via interpolation (should equal the nodal coordinate for affine mapping)
                x_pt = sum(N[a] * node_coords[a, 0] for a in range(4))
                y_pt = sum(N[a] * node_coords[a, 1] for a in range(4))
                # Evaluate material properties at the integration point
                E_val = E_func(x_pt, y_pt)
                nu_val = nu_func(x_pt, y_pt)
                D = constitutive_matrix(E_val, nu_val, analysis_type)
                # Compute stress vector at this point
                stress = D @ (B @ u_e)  # [sigma_x, sigma_y, tau_xy]
                # Accumulate into the global nodal stress for the corresponding global node
                global_node = node_ids[local_idx]
                nodal_stress[global_node, :] += stress
                count[global_node] += 1

    elif elem_type == "tri":
        # For triangular elements (linear), stress is constant over the element.
        for e in range(n_elem):
            node_ids = conn[e, :]
            node_coords = coords[node_ids, :]  # 3x2 array
            dof_ids = []
            for nid in node_ids:
                dof_ids.extend([2 * nid, 2 * nid + 1])
            u_e = U[dof_ids]
            # Compute constant stress (using the centroid; linear triangle yields constant strain)
            A_tri, B_tri = shape_tri(0, 0, node_coords)
            x_c = np.mean(node_coords[:, 0])
            y_c = np.mean(node_coords[:, 1])
            E_val = E_func(x_c, y_c)
            nu_val = nu_func(x_c, y_c)
            D = constitutive_matrix(E_val, nu_val, analysis_type)
            stress = D @ (B_tri @ u_e)
            for local_idx in range(3):
                global_node = node_ids[local_idx]
                nodal_stress[global_node, :] += stress
                count[global_node] += 1

    # Average the accumulated stresses at each node
    for i in range(n_nodes):
        if count[i] > 0:
            nodal_stress[i, :] /= count[i]

    # Compute von Mises stress at each node
    nodal_von_mises = np.sqrt(
        nodal_stress[:, 0] ** 2
        - nodal_stress[:, 0] * nodal_stress[:, 1]
        + nodal_stress[:, 1] ** 2
        + 3 * nodal_stress[:, 2] ** 2
    )

    # --- Deformed mesh and displacement magnitude ---
    deformed_coords = coords + scale * np.column_stack([U[0::2], U[1::2]])
    disp_magnitude = np.sqrt(U[0::2] ** 2 + U[1::2] ** 2)

    # --- Plotting: Use your notebook's exact plotting code (contourf, etc.) ---
    st.header("Results")

    # --- Mesh plot with node classification (from your notebook) ---
    fig = plt.figure(figsize=(Lx, Ly))
    mat_nodes = []
    incl_nodes = []
    dirichlet_nodes = []
    neumann_nodes = []
    for i in range(n_nodes):
        x, y = coords[i]
        in_incl = (incl_x_min - tol <= x <= incl_x_max + tol) and (
            incl_y_min - tol <= y <= incl_y_max + tol
        )
        is_dirichlet = (2 * i in prescribed_dofs) or (2 * i + 1 in prescribed_dofs)
        is_neumann = False
        for bc in bc_list:
            if bc.get("type", None) == "Neumann":
                meets_line = True
                if "line_func" in bc and callable(bc["line_func"]):
                    if abs(bc["line_func"](x, y)) > tol:
                        meets_line = False
                meets_range = True
                if "range" in bc:
                    rng = bc["range"]
                    if "x" in rng:
                        xmin, xmax = rng["x"]
                        if not (xmin - tol <= x <= xmax + tol):
                            meets_range = False
                    if "y" in rng:
                        ymin, ymax = rng["y"]
                        if not (ymin - tol <= y <= ymax + tol):
                            meets_range = False
                if meets_line and meets_range:
                    is_neumann = True
                    break
        if is_dirichlet:
            dirichlet_nodes.append(i)
        elif is_neumann:
            neumann_nodes.append(i)
        else:
            if in_incl:
                incl_nodes.append(i)
            else:
                mat_nodes.append(i)
    if elem_type == "quad":
        for e in range(n_elem):
            node_ids = conn[e, :]
            x_orig = coords[node_ids, 0]
            y_orig = coords[node_ids, 1]
            plt.plot(
                np.append(x_orig, x_orig[0]),
                np.append(y_orig, y_orig[0]),
                "k-",
                linewidth=0.5,
            )
    elif elem_type == "tri":
        for e in range(n_elem):
            node_ids = conn[e, :]
            x_orig = coords[node_ids, 0]
            y_orig = coords[node_ids, 1]
            x_orig = np.append(x_orig, x_orig[0])
            y_orig = np.append(y_orig, y_orig[0])
            plt.plot(x_orig, y_orig, "k-", linewidth=0.5)
    plt.plot(
        coords[mat_nodes, 0],
        coords[mat_nodes, 1],
        "o",
        color="blue",
        label="Matrix Nodes",
        markersize=5,
    )
    plt.plot(
        coords[incl_nodes, 0],
        coords[incl_nodes, 1],
        "o",
        color="red",
        label="Inclusion Nodes",
        markersize=5,
    )
    plt.plot(
        coords[dirichlet_nodes, 0],
        coords[dirichlet_nodes, 1],
        "D",
        color="black",
        label="Dirichlet BC",
        markersize=5,
    )
    plt.plot(
        coords[neumann_nodes, 0],
        coords[neumann_nodes, 1],
        "^",
        color="orange",
        label="Neumann BC",
        markersize=7,
    )
    # traction arrows
    for tr in traction_list:
        for i in range(n_nodes):
            x, y = coords[i]
            if abs(tr["line_func"](x, y)) < tol:
                ok = True
                for dim, rng in tr.get("range", {}).items():
                    val = x if dim == "x" else y
                    if not (rng[0] - tol <= val <= rng[1] + tol):
                        ok = False
                if not ok:
                    continue
                qx, qy = tr["value"](x, y)
                plt.arrow(
                    x,
                    y,
                    0.3 * np.sign(qx),
                    0.3 * np.sign(qy),
                    head_width=0.07,
                    color="black",
                )
    # plt.title("Undeformed Mesh with Node Classification")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend(ncol=4)
    st.pyplot(fig)

    # --- Deformed mesh and stress/disp plots (contourf as in notebook) ---
    ny_nodes = ny_elem + 1
    nx_nodes = nx_elem + 1
    try:
        XX = nodal_stress[:, 0].reshape((ny_nodes, nx_nodes))
        YY = nodal_stress[:, 1].reshape((ny_nodes, nx_nodes))
        XY = nodal_stress[:, 2].reshape((ny_nodes, nx_nodes))
        VM = nodal_von_mises.reshape((ny_nodes, nx_nodes))
        X_def = deformed_coords[:, 0].reshape((ny_nodes, nx_nodes))
        Y_def = deformed_coords[:, 1].reshape((ny_nodes, nx_nodes))
    except Exception:
        XX = YY = XY = VM = X_def = Y_def = None

    if XX is not None:
        # σ_xx
        fig = plt.figure(figsize=(Lx, Ly))
        plt.contourf(X_def, Y_def, XX / 1e6, 200, cmap="jet")
        plt.title(r"$\sigma_{xx}$ on Deformed Mesh")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.colorbar(label="MPa")
        plt.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)
        # σ_yy
        fig = plt.figure(figsize=(Lx, Ly))
        plt.contourf(X_def, Y_def, YY / 1e6, 200, cmap="jet")
        plt.title(r"$\sigma_{yy}$ on Deformed Mesh")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.colorbar(label="MPa")
        plt.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)
        # σ_xy
        fig = plt.figure(figsize=(Lx, Ly))
        plt.contourf(X_def, Y_def, XY / 1e6, 200, cmap="jet")
        plt.title(r"$\sigma_{xy}$ on Deformed Mesh")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.colorbar(label="MPa")
        plt.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)
        # Von Mises
        fig = plt.figure(figsize=(Lx, Ly))
        plt.contourf(X_def, Y_def, VM / 1e6, 200, cmap="jet")
        plt.title("Von Mises Stress on Deformed Mesh")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.colorbar(label="MPa")
        plt.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)

    # Displacement magnitude (scatter, as in notebook)
    fig = plt.figure(figsize=(Lx, Ly))
    plt.scatter(coords[:, 0], coords[:, 1], color="black", label="Original Nodes", s=10)
    sc = plt.scatter(
        deformed_coords[:, 0],
        deformed_coords[:, 1],
        c=disp_magnitude * 1e6,
        cmap="jet",
        label="Deformed Nodes",
        s=20,
    )
    plt.title("Deformed Mesh with Displacement Magnitude (Scale = {})".format(scale))
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.colorbar(sc, label="Displacement (μm)")
    plt.legend(ncol=2)
    plt.axis("equal")
    plt.tight_layout()
    st.pyplot(fig)

    # --- Stress Concentration Factor calculation (from your notebook) ---
    node_principal = np.zeros(n_nodes)
    node_von = np.zeros(n_nodes)
    for i in range(n_nodes):
        sigma_x = nodal_stress[i, 0]
        sigma_y = nodal_stress[i, 1]
        tau_xy = nodal_stress[i, 2]
        p1 = 0.5 * (
            sigma_x + sigma_y + np.sqrt((sigma_x - sigma_y) ** 2 + 4 * tau_xy**2)
        )
        p2 = 0.5 * (
            sigma_x + sigma_y - np.sqrt((sigma_x - sigma_y) ** 2 + 4 * tau_xy**2)
        )
        node_principal[i] = max(p1, p2)
        node_von[i] = np.sqrt(
            sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3 * tau_xy**2
        )
    near_incl = (
        (coords[:, 0] >= incl_x_min - tol)
        & (coords[:, 0] <= incl_x_max + tol)
        & (coords[:, 1] >= incl_y_min - tol)
        & (coords[:, 1] <= incl_y_max + tol)
    )
    max_stress_incl = np.max(node_principal[near_incl])
    mean_stress_far = np.mean(node_principal[~near_incl])
    SCF = max_stress_incl / mean_stress_far
    von_stress_incl = np.max(node_von[near_incl])
    von_stress_far = np.mean(node_von[~near_incl])
    von_SCF = von_stress_incl / von_stress_far

    st.subheader("Stress Concentration Factor (SCF)")
    st.markdown(
        f"**SCF (max principal stress):** {SCF:.3f}  \n"
        f"**SCF (von Mises):** {von_SCF:.3f}"
    )

    # --- Output values ---
    st.subheader("Sample Nodal Displacements (first 10 nodes)")
    st.write(U[:20].reshape(-1, 2))
    st.subheader("Sample Nodal Von Mises Stresses (first 10 nodes)")
    st.write(nodal_von_mises[:10])
    st.success("FEM analysis completed!")
