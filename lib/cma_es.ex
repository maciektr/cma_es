defmodule CmaEs do
  @moduledoc """
  Documentation for `CmaEs`.
  """

  import Nx.Defn

  defstruct [
    :B,
    :C,
    :c1,
    :cc,
    :chiN,
    :cmu,
    :csigma,
    :D,
    :damps,
    :diag_D,
    :dimension,
    :enforce_bounds,
    :fitness_function,
    :initial_solution,
    :initial_step_size,
    :lambda,
    :m,
    :mu,
    :N,
    :p_C,
    :p_sigma,
    :population_size,
    :prev_D,
    :prev_sigma,
    :shape,
    :sigma,
    :weights,
    generation: 0,
    store_trace: false,
    termination_criterion_met: false,
    initialized: false
  ]

  @max_generation 1000
  @termination_no_effect 1.0e-8

  def validate(cma) do
    cond do
      cma.initialized ->
        cma

      cma.initial_solution == nil ->
        raise ArgumentError, "initial_solution is required"

      tuple_size(Nx.shape(cma.initial_solution)) != 1 ->
        raise ArgumentError, "initial_solution must have exactly 1 dimension"

      true ->
        {:ok, cma}
    end
  end

  def init(cma) do
    {:ok, cma} =
      %CmaEs{cma | initialized: false}
      |> validate()

    cma =
      cma
      |> Map.put(:dimension, Nx.shape(cma.initial_solution) |> elem(0))

    n = Nx.tensor(cma.dimension, type: {:f, 32})

    lambda =
      case cma.population_size do
        nil -> Nx.floor(Nx.multiply(Nx.log(n), 3) |> Nx.add(8))
        value -> Nx.tensor(value)
      end

    cma = %{
      cma
      | N: n,
        lambda: lambda,
        shape: Nx.stack([lambda, n]),
        mu: Nx.floor(Nx.divide(lambda, 2)),
        generation: 0,
        initialized: true
    }

    r_heigh = (cma.mu |> Nx.to_flat_list() |> Enum.at(0) |> trunc()) + 1
    range = 1..r_heigh |> Enum.to_list()

    a = Nx.log(Nx.add(cma.mu, 0.5)) |> Nx.subtract(Nx.log(Nx.tensor(range)))

    b =
      Nx.tile(
        Nx.tensor(0, type: {:f, 32}),
        Nx.subtract(lambda, cma.mu) |> Nx.to_flat_list() |> Enum.map(&trunc/1)
      )

    weights =
      Nx.concatenate([
        a,
        b
      ])

    # Normalize weights such as they sum to one and reshape into a column matrix
    weights = Nx.divide(weights, Nx.sum(weights)) |> Nx.reshape({:auto, 1})

    # Variance-effective size of mu
    mueff = Nx.sum(weights) |> Nx.power(2) |> Nx.divide(Nx.sum(Nx.power(weights, 2)))

    cc =
      case cma.cc do
        nil ->
          Nx.add(4, Nx.divide(mueff, Map.get(cma, :N)))
          |> Nx.divide(
            Nx.add(Map.get(cma, :N), 4)
            |> Nx.add(Nx.multiply(2, Nx.divide(mueff, Map.get(cma, :N))))
          )

        value ->
          Nx.tensor(value)
      end

    csigma =
      case cma.csigma do
        nil -> Nx.add(mueff, 2) |> Nx.divide(Nx.add(Map.get(cma, :N), Nx.add(mueff, 5)))
        value -> Nx.tensor(value)
      end

    c1 =
      case cma.c1 do
        nil -> Nx.divide(2, Nx.power(Nx.add(Map.get(cma, :N), 1.3), 2) |> Nx.add(mueff))
        value -> Nx.tensor(value)
      end

    # # Learning rate for rank-μ update of C
    cmu =
      case cma.cmu do
        nil ->
          Nx.multiply(2, Nx.subtract(mueff, 2) |> Nx.add(Nx.divide(1, mueff)))
          |> Nx.divide(
            Nx.power(Nx.add(Map.get(cma, :N), 2), 2)
            |> Nx.add(Nx.divide(Nx.multiply(2, mueff), 2))
          )

        value ->
          Nx.tensor(value)
      end

    # # Damping for sigma

    damps =
      case cma.damps do
        nil ->
          2
          |> Nx.multiply(
            Nx.max(
              0,
              Nx.subtract(
                Nx.sqrt(Nx.subtract(mueff, 1) |> Nx.divide(Nx.add(Map.get(cma, :N), 1))),
                1
              )
            )
          )
          |> Nx.add(1)
          |> Nx.add(csigma)

        value ->
          Nx.tensor(value)
      end

    # # Expectation of ||N(0,I)||
    chiN =
      Nx.sqrt(Map.get(cma, :N))
      |> Nx.multiply(
        Nx.subtract(1, Nx.divide(1, Nx.multiply(4, Map.get(cma, :N))))
        |> Nx.add(Nx.divide(1, Nx.multiply(21, Nx.power(Map.get(cma, :N), 2))))
      )

    # # Define bounds in a format that can be fed to tf.clip_by_value

    # # Trainable parameters
    # # Mean
    m = cma.initial_solution
    # # Step-size
    sigma = Nx.tensor(cma.initial_step_size)
    # # Covariance matrix
    s = Map.get(cma, :N) |> Nx.to_flat_list() |> Enum.map(&trunc/1) |> Enum.at(0)
    c = Nx.eye({1, s})
    # # Evolution path for σ
    p_sigma = Nx.tile(Nx.tensor(0), [s])
    # # Evolution path for C
    p_C = Nx.tile(Nx.tensor(0), [s])
    # # Coordinate system (normalized eigenvectors)
    b = Nx.eye({1, s})
    # # Scaling (square root of eigenvalues)
    d = Nx.eye({1, s})

    %{
      cma
      | weights: weights,
        cc: cc,
        csigma: csigma,
        c1: c1,
        cmu: cmu,
        damps: damps,
        chiN: chiN,
        m: m,
        sigma: sigma,
        C: c,
        p_sigma: p_sigma,
        p_C: p_C,
        B: b,
        D: d
    }
  end

  def step(cma) do
    # 1 Sample a new population
    z = Nx.random_normal(cma.shape)
    y = Nx.dot(z, Nx.dot(Map.get(cma, :B), Map.get(cma, :D)))
    x = Nx.add(Map.get(cma, :m), Nx.multiply(Map.get(cma, :sigma), y))

    penalty = 0

    # 2 Selection and Recombination: Moving the Mean
    f_x = cma.fitness_fn.(x) + penalty
    x_sorted = Nx.gather(x, Nx.argsort(f_x))
    # The new mean is a weighted average of the top-μ solutions
    x_diff = Nx.subtract(x_sorted, Map.get(cma, :m))
    x_mean = Nx.sum(Nx.multiply(x_diff, Map.get(cma, :weights)), axes: [0])
    m = Nx.add(Map.get(cma, :m), x_mean)

    # 3 Adapting the Covariance Matrix
    # # Udpdate evolution path for Rank-one-Update
    y_mean = Nx.divide(x_mean, Map.get(cma, :sigma))

    p_C =
      Nx.add(
        Nx.multiply(Nx.subtract(1, Map.get(cma, :cc)), Map.get(cma, :p_C)),
        Nx.multiply(
          Nx.sqrt(
            Nx.multiply(
              Nx.multiply(Map.get(cma, :cc), Nx.subtract(2, Map.get(cma, :cc))),
              Map.get(cma, :μeff)
            )
          ),
          y_mean
        )
      )

    p_C_matrix = Nx.expand_dims(p_C, axis: 1)

    # # Compute Rank-μ-Update
    C_m =
      Nx.map(Nx.expand_dims(Nx.divide(x_diff, Map.get(cma, :sigma)), axis: 1), fn e ->
        Nx.dot(e, Nx.transpose(e))
      end)

    y_s = Nx.sum(Nx.multiply(C_m, Nx.expand_dims(Map.get(cma, :weights), axis: 1)), axes: [0])

    # # Combine Rank-one-Update and Rank-μ-Update
    C =
      Nx.add(
        Nx.add(
          Nx.multiply(
            Nx.subtract(Nx.subtract(1, Map.get(cma, :c1)), Map.get(cma, :cmu)),
            Map.get(cma, :C)
          ),
          Nx.multiply(Map.get(cma, :c1), Nx.dot(p_C_matrix, Nx.transpose(p_C_matrix)))
        ),
        Nx.multiply(Map.get(cma, :cmu), y_s)
      )

    # # Enforce symmetry of the covariance matrix
    C_upper = Nx.band_part(C, 0, -1)
    C_upper_no_diag = Nx.subtract(C_upper, Nx.diag_part(C_upper))
    C = Nx.add(C_upper, Nx.transpose(C_upper_no_diag))

    # 4 Step-size control
    # #Update evolution path for sigma
    D_inv = Nx.diag_part(Map.get(cma, :D))
    C_inv_squared = Nx.dot(Nx.dot(Map.get(cma, :B), D_inv), Nx.transpose(Map.get(cma, :B)))
    C_inv_squared_y = Nx.squeeze(Nx.dot(C_inv_squared, Nx.expand_dims(y_mean, axis: 1)))

    p_sigma =
      Nx.add(
        Nx.multiply(Nx.subtract(1, Map.get(cma, :cs)), Map.get(cma, :p_sigma)),
        Nx.multiply(
          Nx.sqrt(
            Nx.multiply(
              Nx.multiply(Map.get(cma, :cs), Nx.subtract(2, Map.get(cma, :cs))),
              Map.get(cma, :μeff)
            )
          ),
          C_inv_squared_y
        )
      )

    # # Update sigma
    sigma =
      Map.get(cma, :sigma) *
        Nx.exp(
          Nx.divide(
            Nx.multiply(
              Nx.divide(Map.get(cma, :cs), Map.get(cma, :damps)),
              Nx.subtract(Nx.divide(Nx.norm(p_sigma), Map.get(cma, :chiN)), 1)
            )
          )
        )

    # 5 Update B and D: eigen decomposition
    {u, B, _} = Nx.LinAlg.svd(C)

    # diag_D = tf.sqrt(u)
    diag_D = Nx.sqrt(u)
    D = Nx.diag_part(diag_D)

    # 6 Assign new variable values
    cma = %CmaEs{
      cma
      | generation: cma.generation + 1,
        prev_sigma: Nx.tensor(cma.sigma),
        prev_D: Nx.tensor(Map.get(cma, :D)),
        diag_D: diag_D,
        p_C: p_C,
        p_sigma: p_sigma,
        C: C,
        sigma: sigma,
        B: B,
        D: D,
        m: m
    }

    {:ok, cma}
  end

  def search(cma, 0) do
    {:ok, cma}
  end

  def search(cma, steps) do
    case cma |> step do
      {:ok, next_cma} -> search(next_cma, steps - 1)
      _ -> {:ok, cma}
    end
  end

  @spec search(CmaEs) :: {:ok, CmaEs}
  def search(cma) do
    search(cma, @max_generation)
  end
end
