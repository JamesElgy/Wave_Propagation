import wave_propagation as wp

if __name__ == '__main__':

    wp.Testing.scipy_random_solve()

    # wp.Testing.run_p_ref(solver='CG', preconditioner='direct', tol=1e-12)
    wp.Testing.run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12)
    # wp.Testing.run_p_ref(solver='scipy', preconditioner='bddc', tol=1e-12)
    # wp.Testing.run_p_ref(solver='scipy', preconditioner='multigrid', tol=1e-12)

    # W = wp.wave_propagation()
    # W.run()