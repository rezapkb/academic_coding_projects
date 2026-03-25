# This code finds e-values and e-vecs of the tetrahedron problem for given input parameters j1, j2, j3, j4
# and the finds 7 terms in the thermal density.
# and includes a routine for calculating the concurrence.

from scipy.optimize import linprog
import numpy as np
import sys
import os

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

# ----- Sweep setup -----
N = 180
vals = np.linspace(-1, 1, N)

J1_list = [+1.0, -1.0]
T_list = [0.01, 0.1, 0.2, 2.0, 10.0]

pair_sweeps = [
    ("J2", "J3"),
    ("J3", "J4"),
    ("J2", "J4"),
]

# storage bank:
# conc_bank[(pair, J1, T)] = concurrence map
conc_bank = {}

optimized_conc_map = np.zeros((N, N))

jobs = []
for px, py in pair_sweeps:
    for J1_fixed in J1_list:
        for T_fixed in T_list:
            jobs.append((px, py, J1_fixed, T_fixed))

px, py, J1_fixed, T_fixed = jobs[task_id]


for i, xval in enumerate(vals):
    for j, yval in enumerate(vals):

        # default fixed values
        j1 = J1_fixed
        j2 = 0.0
        j3 = 0.0
        j4 = 0.0
        T = T_fixed

        if px == "J2":
            j2 = xval
        if px == "J3":
            j3 = xval
        if px == "J4":
            j4 = xval

        if py == "J2":
            j2 = yval
        if py == "J3":
            j3 = yval
        if py == "J4":
            j4 = yval

        ep = -0.5 + 1j * np.sqrt(3) / 2
        ep2 = -0.5 - 1j * np.sqrt(3) / 2
        # symmetrized states
        A1 = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 1,
                       1, 0, 1, 0, 0, 0]) / np.sqrt(6)
        Eplus3 = np.array(
            [0, 0, 0, 1, 0, ep, ep2, 0, 0, ep2, ep, 0, 1, 0, 0, 0]
        ) / np.sqrt(6)
        Eminus3 = np.conj(Eplus3)
        T1x2 = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2)
        T1y2 = np.array(
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]) / np.sqrt(2)
        T1z2 = np.array(
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]) / np.sqrt(2)
        Eplus1 = np.array(
            [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        Eminus1 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
        Eplus2 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 0.5, 0, 0.5, 0.5, 0])
        Eminus2 = np.array(
            [0, -1 / 2, -1 / 2, 0, -1 / 2, 0, 0,
             0, -1 / 2.0, 0, 0, 0, 0, 0, 0, 0]
        )
        T1x1 = np.array(
            [0, -ep2, ep2, 0, ep2, 0, 0, -ep, -
             ep2, 0, 0, ep, 0, ep, -ep, 0]
        ) / (2 * np.sqrt(2))
        T1y1 = np.array(
            [0, ep, -ep, 0, ep, 0, 0, -ep2, -
             ep, 0, 0, ep2, 0, -ep2, ep2, 0]
        ) / (2 * np.sqrt(2))
        T1z1 = np.array([0, 1, 1, 0, -1, 0, 0, -1, -1, 0,
                         0, -1, 0, 1, 1, 0]) / (2 * np.sqrt(2))
        T2x = np.array(
            [0, -ep2, ep2, 0, ep2, 0, 0, ep, -
             ep2, 0, 0, -ep, 0, -ep, ep, 0]
        ) / (2 * np.sqrt(2))
        T2y = np.array(
            [0, ep, -ep, 0, ep, 0, 0, ep2, -
             ep, 0, 0, -ep2, 0, ep2, -ep2, 0]
        ) / (2 * np.sqrt(2))
        T2z = np.array([0, 1, 1, 0, -1, 0, 0, 1, -1, 0, 0,
                        1, 0, -1, -1, 0]) / (2 * np.sqrt(2))
        sym = [
            A1,
            Eplus1,
            Eplus2,
            Eplus3,
            Eminus1,
            Eminus2,
            Eminus3,
            T1x1,
            T1x2,
            T1y1,
            T1y2,
            T1z1,
            T1z2,
            T2x,
            T2y,
            T2z,
        ]

        # check orthogonality
        # test = np.zeros((16,16), dtype = "complex")
        # for count1 in range (0,16):
        #   for count2 in range (0,16):
        #      test[count1][count2] = np.dot(sym[count1], np.conj(sym[count2]))
        # print(test)

        # The seven e-values
        a_eval = j1 / 6 - 2 * j4 / 3
        ematrix = [
            [-j1 / 2, 0, np.sqrt(2 / 3) * j3],
            [0, -j4 / 2, -2 * j2 / np.sqrt(3)],
            [np.sqrt(2 / 3) * j3, -2 * j2 /
             np.sqrt(3), j1 / 6 + j4 / 3],
        ]
        e_eval, e_evec = np.linalg.eig(ematrix)

        # Sort eigenvalues and reorder eigenvectors accordingly
        idx = np.argsort(e_eval.real)
        e_eval = e_eval.real[idx]
        e_evec = e_evec[:, idx]
        tmatrix = [
            [2 * j3 / 3 + j4 / 6, 2 * np.sqrt(2) * j2 / 3],
            [2 * np.sqrt(2) * j2 / 3, j1 / 6],
        ]
        t1_eval, t1_evec = np.linalg.eig(tmatrix)

        idx_t = np.argsort(t1_eval.real)
        t1_eval = t1_eval.real[idx_t]
        t1_evec = t1_evec[:, idx_t]
        t2_eval = -2 * j3 / 3 + j4 / 6
        evals = [
            a_eval,
            e_eval[0],
            e_eval[1],
            e_eval[2],
            t1_eval[0],
            t1_eval[1],
            t2_eval,
        ]

        degens = [1, 2, 2, 2, 3, 3, 3]

        # the partition function
        Z_part = 0
        for count in range(0, 7):
            Z_part += degens[count] * np.exp(-evals[count] / T)

            # probabilities for initial densities
            p_init = np.zeros(11, dtype=float)
            for count in range(0, 7):
                p_init[count] = np.exp(-evals[count] / T) / Z_part
            # numerical safety
            p_init = np.clip(p_init, 0.0, 1.0)

            # print(Z_part, p_init)

            def concur(state):
                bin_state = np.resize(state, (2, 2, 2, 2))
                conc = 0
                for a in range(0, 2):
                    for b in range(0, 2):
                        for c in range(0, 2):
                            for d in range(0, 2):
                                for e in range(0, 2):
                                    for f in range(0, 2):
                                        for g in range(0, 2):
                                            for h in range(0, 2):
                                                # First four are
                                                # the 1-part
                                                # reduced densities
                                                conc += (
                                                    bin_state[a][b][c][d] *
                                                    np.conj(
                                                        bin_state[a][e][f][g]) *
                                                    bin_state[h][e][f][g] *
                                                    np.conj(
                                                        bin_state[h][b][c][d]))
                                            conc += (
                                                bin_state[a][b][c][d] *
                                                np.conj(
                                                    bin_state[e][b][f][g]) *
                                                bin_state[e][h][f][g] *
                                                np.conj(
                                                    bin_state[a][h][c][d]))
                                            conc += (
                                                bin_state[a][b][c][d] *
                                                np.conj(
                                                    bin_state[e][f][c][g]) *
                                                bin_state[e][f][h][g] *
                                                np.conj(
                                                    bin_state[a][b][h][d]))
                                            conc += (
                                                bin_state[a][b][c][d] *
                                                np.conj(
                                                    bin_state[e][f][g][d]) *
                                                bin_state[e][f][g][h] *
                                                np.conj(
                                                    bin_state[a][b][c][h]))
                                            # next 6 are 2-part
                                            # reduced densities
                                            conc += (
                                                bin_state[a][b][c][d] *
                                                np.conj(
                                                    bin_state[a][b][e][f]) *
                                                bin_state[g][h][e][f] *
                                                np.conj(
                                                    bin_state[g][h][c][d]))
                                            conc += (
                                                bin_state[a][b][c][d] *
                                                np.conj(
                                                    bin_state[a][e][c][f]) *
                                                bin_state[g][e][h][f] *
                                                np.conj(
                                                    bin_state[g][b][h][d]))
                                            conc += (
                                                bin_state[a][b][c][d] *
                                                np.conj(
                                                    bin_state[a][e][f][d]) *
                                                bin_state[g][e][f][h] *
                                                np.conj(
                                                    bin_state[g][b][c][h]))
                conc = 2 * np.real(conc)
                conc = np.sqrt(14 - conc) / 2
                return conc

            # print("E-concur", concur((Eplus1+Eplus2)/np.sqrt(2)))
            # for count in range (0,16):
            # print(count, concur(sym[count]))

            # This function extracts the elements of a density
            # array for a given state

            def dens_array(state):
                dens = np.zeros((11), dtype=float)
                dens[0] = np.real(state[0] * np.conj(state[0]))
                dens[1] = np.real(state[1] * np.conj(state[1]))
                dens[2] = np.real(state[3] * np.conj(state[3]))
                dens[3] = np.real(state[10] * np.conj(state[5]))
                dens[4] = np.real(state[3] * np.conj(state[0]))
                dens[5] = np.real(state[7] * np.conj(state[0]))
                dens[6] = np.real(state[2] * np.conj(state[1]))
                dens[7] = np.real(state[3] * np.conj(state[2]))
                dens[8] = np.real(state[12] * np.conj(state[1]))
                dens[9] = np.real(state[13] * np.conj(state[1]))
                dens[10] = np.real(state[5] * np.conj(state[3]))
                return dens

            # for count in range (0,16):
            # print(dens_array(sym[count]))

            # print(Eplus1)
            # print(e_evec)

            # Here we find the 7 initial density arrays.
            dens_init = np.zeros((7, 11), dtype=float)
            #  A1 state
            dens_init[0] = dens_array(A1)
            # E states
            # E_evecplus1 = Eplus1
            E_evecplus1 = (
                e_evec[0][0] * Eplus1 + e_evec[0][1] *
                Eplus2 + e_evec[0][2] * Eplus3
            )
            E_evecplus2 = (
                e_evec[1][0] * Eplus1 + e_evec[1][1] *
                Eplus2 + e_evec[1][2] * Eplus3
            )
            E_evecplus3 = (
                e_evec[2][0] * Eplus1 + e_evec[2][1] *
                Eplus2 + e_evec[2][2] * Eplus3
            )
            E_evecminus1 = (
                e_evec[0][0] * Eminus1 + e_evec[0][1] *
                Eminus2 + e_evec[0][2] * Eminus3
            )
            E_evecminus2 = (
                e_evec[1][0] * Eminus1 + e_evec[1][1] *
                Eminus2 + e_evec[1][2] * Eminus3
            )
            E_evecminus3 = (
                e_evec[2][0] * Eminus1 + e_evec[2][1] *
                Eminus2 + e_evec[2][2] * Eminus3
            )
            dens_init[1] = (dens_array(E_evecplus1) +
                            dens_array(E_evecminus1)) / 2.0
            dens_init[2] = (dens_array(E_evecplus2) +
                            dens_array(E_evecminus2)) / 2.0
            dens_init[3] = (dens_array(E_evecplus3) +
                            dens_array(E_evecminus3)) / 2.0
            T1_evecx1 = t1_evec[0][0] * T1x1 + t1_evec[0][1] * T1x2
            T1_evecy1 = t1_evec[0][0] * T1y1 + t1_evec[0][1] * T1y2
            T1_evecz1 = t1_evec[0][0] * T1z1 + t1_evec[0][1] * T1z2
            T1_evecx2 = t1_evec[1][0] * T1x1 + t1_evec[1][1] * T1x2
            T1_evecy2 = t1_evec[1][0] * T1y1 + t1_evec[1][1] * T1y2
            T1_evecz2 = t1_evec[1][0] * T1z1 + t1_evec[1][1] * T1z2
            dens_init[4] = (
                dens_array(
                    T1_evecx1) + dens_array(T1_evecy1) + dens_array(T1_evecz1)
            ) / 3.0
            dens_init[5] = (
                dens_array(
                    T1_evecx2) + dens_array(T1_evecy2) + dens_array(T1_evecz2)
            ) / 3.0
            dens_init[6] = (dens_array(T2x) +
                            dens_array(T2y) + dens_array(T2z)) / 3

            # print("dens_init", dens_init)

            # put the concurrences in an array of size 7, and pad
            # with 4 more zeros
            concurrence = np.zeros((11))
            concurrence[0] = concur(A1)
            concurrence[1] = concur(E_evecplus1)
            concurrence[2] = concur(E_evecplus2)
            concurrence[3] = concur(E_evecplus3)
            concurrence[4] = concur(T1_evecx1)
            concurrence[5] = concur(T1_evecx2)
            concurrence[6] = concur(T2x)

            # print("concur", concurrence, "p_intit", p_init)
            # We should calculate the concurrence of our starting
            # point
            c_init = np.dot(concurrence, p_init)
            # print("c_init", c_init)

            # Now we import the 67-some unentangled densities
            # These were found using my sites_operations4.0.py and Deep's jyputer notebook that removes those that are
            # dependent on others as a positive sum.
            # we started with 1296 density_arrays
            # of which 99 were unique
            # and 67 are independent

            dens_unentang = np.loadtxt("independent.tsv")
            dens_unentang = dens_unentang / 24

            # print("dens_unentang", dens_unentang)

            from scipy.optimize import linprog

            # Select 4 of the unentangled vectors that are independent to the dens_init vectors, and to each other.
            # the set of 11 independent vecs will be called dens_independent
            # put the remaining unentangled vectors into a smaller
            # set  called dens_unentang_left
            dens_independent = np.zeros((11, 11))
            dens_unentang_left = np.zeros((63, 11))
            for count in range(0, 7):
                dens_independent[count] = dens_init[count]
                # Just for testing - including these should eliminate them  - yes, it does
                # for count in range (7,11):
                #   dens_independent[count] = dens_unentang[count-7]
                # If eta_vec is to be checked, look for a  solution to
                # eta_vec = sum_i  x_i
                chosen_vec_number = 7
                left_over_number = 0
                for count in range(0, 67):
                    b_eq = dens_unentang[count]
                    A_eq = dens_independent
                    c = np.zeros((11))
                    result = linprog(
                        c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None))
                    if result.success:
                        dens_unentang_left[left_over_number] = dens_unentang[count]
                        left_over_number += 1
                       # print("success")
                    else:
                        dens_independent[chosen_vec_number] = b_eq
                        chosen_vec_number += 1
                       # print("False - found an independent vec")
                   # print(count, chosen_vec_number)
                    if chosen_vec_number == 11:
                        for count3 in range(count + 1, 67):
                            dens_unentang_left[left_over_number] = dens_unentang[count3]
                            left_over_number += 1
                        break
               # print("dens1", dens_independent)
               # print(left_over_number, dens_unentang_left)
               # print("det1", np.linalg.det(dens_independent))

               # Now we define the 11X11 matrix used to re-express the
               # remaining unentangled densities
                u_inverse = np.linalg.pinv(dens_independent)
                dens_unentang_left_new = np.zeros((63, 11))
                for count in range(0, 63):
                    dens_unentang_left_new[count] = np.matmul(
                        dens_unentang_left[count], u_inverse
                    )
               # print("dens_unentang_left_new", dens_unentang_left_new)
                # This matrix has 63 rows and 11 columns.  Each row corresponds to an unentangled density u.
                # The elements in each row are the coefficients a_i
                # where u = sum_i a_i v_i, and v_i is an independent
                # density.

                # Do the optimization problem to find the concurrence
                # need to minimize the function c(x) = c_init - sum_j
                # x_j a_ji c_i
                c = np.zeros((63), dtype=float)
                for count in range(0, 63):
                    c[count] = - \
                        np.dot(
                            dens_unentang_left_new[count], concurrence)
                # print("c",c)

                # The various connstraints
                # There are a total of (11+63)X2 = 148 constraints on
                # the 63 x's
                A_ub = np.zeros((148, 63), dtype=float)
                b_ub = np.zeros((148), dtype=float)
                # The first set of constraints are x_i < 1
                Aub1 = np.identity(63)
                bub1 = np.ones((63))
                # The next set is -x_i < 0
                Aub2 = -np.identity(63)
                bub2 = np.zeros(63)
                # The next set is p_i - alpha_{ji} x_j < 1  or  -
                # alpha_{ji} x_j < 1 -p_i
                Aub3 = -np.transpose(dens_unentang_left_new)
                bub3 = 1 - p_init
                # The final set is 0 < p_i - alpha_{ji} x_j  or
                # alpha_{ji} x_j < p_i
                Aub4 = np.transpose(dens_unentang_left_new)
                bub4 = p_init

                # print("Aub1", Aub1, bub1)
                # print("Aub2", Aub2)
                # print( dens_unentang_left_new)
                # print("Aub3", Aub3, bub3)
                # print("Aub4", Aub4, bub4)

                for count in range(0, 63):
                    A_ub[count] = Aub1[count]
                    A_ub[count + 63] = Aub2[count]
                    b_ub[count] = bub1[count]
                    b_ub[count + 63] = bub2[count]
                for count in range(0, 11):
                    count2 = count + 63 * 2
                    A_ub[count2] = Aub3[count]
                    A_ub[count2 + 11] = Aub4[count]
                    b_ub[count2] = bub3[count]
                    b_ub[count2 + 11] = bub4[count]

                # print(c, A_ub, b_ub)

                p_final = np.zeros((11 + 63), dtype=float)
                result = linprog(c, A_ub, b_ub, A_eq=None,
                                 b_eq=None, bounds=(0, None))
                # --- Add normalization equality constraint ---
                A_eq = np.sum(dens_unentang_left_new, axis=1).reshape(1, -1)
                b_eq = np.array([0.0])

                result = linprog(
                    c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=(0, None),
                    method="highs"
                )

                # print(result)

                # Check if optimization succeeded
                if result.success:
                    # print(c_init, c_init + result.fun, result.fun)
                    # store optimized concurrence
                    c_final = c_init + result.fun
                else:
                    # print("Optimization failed:", result.message)
                    # Decide what to do if it fails:
                    # For example, set c_final = c_init
                    c_final = c_init

                optimized_conc_map[j, i] = c_final

                np.save(
                    f"conc_{px}_{py}_J1_{J1_fixed}_T_{T_fixed}.npy",
                    optimized_conc_map
                )
