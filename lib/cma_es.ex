defmodule CmaEs do
  @moduledoc """
  Documentation for `CmaEs`.
  """

  # import Nx.Defn

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
    :mueff,
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

  # @max_generation 1000
  @max_generation 100
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

    n = Nx.tensor(cma.dimension, type: {:s, 32})

    lambda =
      case cma.population_size do
        nil -> Nx.floor(Nx.multiply(Nx.log(n), 3) |> Nx.add(8))
        value -> Nx.tensor(value, type: {:s, 32})
      end

    cma = %{
      cma
      | N: n,
        lambda: lambda,
        shape: Nx.stack([lambda, n]) |> Nx.as_type({:s, 32}),
        mu: Nx.floor(Nx.divide(lambda, 2)),
        generation: 0,
        initialized: true
    }

    r_heigh = cma.mu |> Nx.to_flat_list() |> Enum.at(0) |> trunc()
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
    b = Nx.eye(s)
    # # Scaling (square root of eigenvalues)
    d = Nx.eye(s)

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
        D: d,
        mueff: mueff
    }
  end

  def step(cma) do
    # 1 Sample a new population
    # z = Nx.random_normal(cma.shape)
    # key = Nx.Random.key(42)
    # {z, key} = Nx.Random.normal(key, 0, 1, shape: cma.shape |> Nx.to_flat_list() |> List.to_tuple(), type: :f32)
    shape = cma.shape |> Nx.to_flat_list() |> List.to_tuple()
    z = Nx.random_normal(shape)
    # cma.shape |> IO.puts(Nx.shape(z), z)
    # tmp = Nx.dot(Map.get(cma, :B), Map.get(cma, :D))
    # IO.inspect({:z, z, Map.get(cma, :B), Map.get(cma, :D)})
    tmp = Nx.dot(Map.get(cma, :B), Map.get(cma, :D))
    # IO.inspect({:tmp, tmp})
    y = Nx.dot(z, tmp)
    # IO.inspect({:z, z, Nx.multiply(cma.sigma, y)})
    x = Nx.add(cma.m, Nx.multiply(cma.sigma, y))
    # IO.inspect({:x, x})
    penalty = 0

    # 2 Selection and Recombination: Moving the Mean
    f_x = cma.fitness_function.(x) |> Nx.add(penalty)
    # |> Nx.new_axis(1)
    tmp = Nx.argsort(f_x)
    tl = Nx.shape(x) |> elem(1)
    tmp = Nx.stack(for(_ <- 1..tl, do: tmp), axis: 1)
    # tmp = Nx.stack([tmp,tmp]) |> Nx.reshape({Nx.shape(tmp) |> elem(0), 2})
    # tmp_dim = Nx.shape(tmp) |> elem(0)
    # tmp = tmp |> Nx.reshape({tmp_dim, 1})
    # tmp_zeros = Nx.tensor(for _ <- 1..tmp_dim, do: 0)
    # tmp = Nx.stack([tmp, tmp_zeros], axis: 1)

    # x_sorted = Nx.gather(x, tmp)
    # x_sorted = Nx.new_axis(x_sorted, 1)
    x_sorted = Nx.take_along_axis(x, tmp, axis: 0)
    # IO.inspect({:x_sorted, x_sorted, tmp})

    # The new mean is a weighted average of the top-μ solutions
    x_diff = Nx.subtract(x_sorted, cma.m)
    # IO.inspect({:x_diff, x_diff, cma.m})
    x_mean = Nx.sum(Nx.multiply(x_diff, cma.weights), axes: [0])
    # IO.inspect({:x_mean, x_mean})
    m = Nx.add(cma.m, x_mean)

    # 3 Adapting the Covariance Matrix
    # # Udpdate evolution path for Rank-one-Update
    y_mean = Nx.divide(x_mean, Map.get(cma, :sigma))

    p_C =
      Nx.subtract(1, Map.get(cma, :cc))
      |> Nx.multiply(Map.get(cma, :p_C))
      |> Nx.add(
        Map.get(cma, :cc)
        |> Nx.multiply(Nx.subtract(2, Map.get(cma, :cc)))
        |> Nx.multiply(Map.get(cma, :mueff))
        |> Nx.sqrt()
        |> Nx.multiply(y_mean)
      )

    # p_C_matrix = Nx.expand_dims(p_C, axis: 1)
    p_C_matrix = Nx.new_axis(p_C, Nx.shape(p_C) |> tuple_size())

    # # Compute Rank-μ-Update
    tmp = Nx.divide(x_diff, Map.get(cma, :sigma))

    to_map = Nx.new_axis(tmp, Nx.shape(tmp) |> tuple_size())

    c_m =
      Nx.map(to_map, fn e ->
        Nx.dot(e, Nx.transpose(e))
      end)

    tmp = Map.get(cma, :weights)

    y_s =
      Nx.sum(
        Nx.multiply(
          c_m,
          # Nx.expand_dims(Map.get(cma, :weights), axis: 1)
          Nx.new_axis(tmp, Nx.shape(tmp) |> tuple_size())
        ),
        axes: [0]
      )

    # # Combine Rank-one-Update and Rank-μ-Update

    c =
      Nx.subtract(1, Map.get(cma, :c1))
      |> Nx.subtract(Map.get(cma, :cmu))
      |> Nx.multiply(Map.get(cma, :C))
      |> Nx.add(Nx.multiply(Map.get(cma, :c1), Nx.dot(p_C_matrix, Nx.transpose(p_C_matrix))))
      |> Nx.add(Nx.multiply(Map.get(cma, :cmu), y_s))

    # # Enforce symmetry of the covariance matrix
    # c_upper = Nx.band_part(c, 0, -1)
    {tx, _ty} = Nx.shape(c)
    # ind = for r <- 0..tx-1, k <- 0..ty-1 do
    #     if k >= r do
    #       [r, k]
    #     end
    # end |> Enum.filter(& !is_nil(&1))
    # c_upper = Nx.tensor(ind)
    # c_upper = Nx.gather(c, Nx.tensor(ind))
    c_upper = c
    # for r <- 0..tx-1 do
    #   c_upper = Nx.put_slice(c_upper, [r,0], Nx.tensor(for _ <- 0..r, do: 0))
    # end

    c_upper =
      Enum.reduce(0..(tx - 1), c_upper, fn r, acc ->
        if r == 0 do
          acc
        else
          Nx.put_slice(acc, [r, 0], Nx.tensor([for(_ <- 0..(r - 1), do: 0)]))
        end
      end)

    # c_upper_no_diag = Nx.subtract(c_upper, Nx.diag_part(c_upper))
    c_upper_no_diag = Nx.subtract(c_upper, Nx.take_diagonal(c_upper))
    c = Nx.add(c_upper, Nx.transpose(c_upper_no_diag))

    # 4 Step-size control
    # #Update evolution path for sigma
    # d_inv = Nx.diag_part(Map.get(cma, :D))
    d_inv = Nx.take_diagonal(Map.get(cma, :D))
    # c_inv_squared = Nx.dot(Nx.dot(Map.get(cma, :B), d_inv), Nx.transpose(Map.get(cma, :B)))
    #!!!1 c_inv_squared = Nx.dot(Nx.dot(Map.get(cma, :B), d_inv), Nx.transpose(Map.get(cma, :B)))
    c_inv_squared =
      Nx.multiply(Nx.multiply(Map.get(cma, :B), d_inv), Nx.transpose(Map.get(cma, :B)))

    y_mean_expanded = Nx.new_axis(y_mean, Nx.shape(y_mean) |> tuple_size())
    # c_inv_squared_y = Nx.squeeze(Nx.dot(c_inv_squared, y_mean_expanded      ))
    #!!! c_inv_squared_y = Nx.squeeze(Nx.dot(y_mean_expanded, c_inv_squared))
    c_inv_squared_y = Nx.squeeze(Nx.multiply(y_mean_expanded, c_inv_squared))

    p_sigma =
      Nx.add(
        Nx.multiply(Nx.subtract(1, Map.get(cma, :csigma)), Map.get(cma, :p_sigma)),
        Nx.multiply(
          Nx.sqrt(
            Nx.multiply(
              Nx.multiply(Map.get(cma, :csigma), Nx.subtract(2, Map.get(cma, :csigma))),
              Map.get(cma, :mueff)
            )
          ),
          c_inv_squared_y
        )
      )

    #   IO.inspect({:chin, Map.get(cma, :chiN), p_sigma})
    #   IO.inspect({:lin, Nx.LinAlg.norm(p_sigma)})
    #   IO.inspect(
    # Nx.divide(Nx.LinAlg.norm(p_sigma), Map.get(cma, :chiN))
    # )
    #       Nx.multiply(
    #         Nx.divide(Map.get(cma, :csigma), Map.get(cma, :damps)),
    #         Nx.subtract(Nx.divide(Nx.LinAlg.norm(p_sigma), Map.get(cma, :chiN)), 1)
    #       )
    # )
    # # Update sigma
    sigma =
      Map.get(cma, :sigma)
      |> Nx.multiply(
        Nx.exp(
          Nx.multiply(
            Nx.divide(Map.get(cma, :csigma), Map.get(cma, :damps)),
            Nx.subtract(Nx.divide(Nx.LinAlg.norm(p_sigma), Map.get(cma, :chiN)), 1)
          )
        )
      )

    # IO.inspect({:sigm, sigma |> Nx.to_flat_list() |> List.first(), Map.get(cma, :sigma)|> Nx.to_flat_list() |> List.first()})

    # 5 Update B and D: eigen decomposition
    # {u, b, _} = Nx.LinAlg.svd(c)
    {b, u, _} = Nx.LinAlg.svd(c)

    # diag_D = tf.sqrt(u)
    # IO.inspect({:u, u, c})

    diag_D =
      Nx.map(u, fn e ->
        Nx.sqrt(Nx.max(e, Nx.tensor(0.0)))
      end)

    # diag_D = Nx.sqrt(u)
    # d = Nx.diag_part(diag_D)
    # d = Nx.take_diagonal(diag_D)
    # IO.inspect({:diag, diag_D})
    dl = Nx.shape(diag_D) |> elem(0)
    d = Nx.broadcast(0, {dl, dl}) |> Nx.put_diagonal(diag_D |> Nx.flatten())
    # IO.inspect({:d, d})

    # 6 Assign new variable values
    cma = %CmaEs{
      cma
      | generation: cma.generation + 1,
        prev_sigma: cma.sigma,
        prev_D: Map.get(cma, :D),
        diag_D: diag_D,
        p_C: p_C,
        p_sigma: p_sigma,
        C: c,
        sigma: sigma,
        B: b,
        D: d,
        m: m
    }

    # IO.inspect({cma.generation, m})
    {:ok, cma}
  end

  def search(cma, 0) do
    {:ok, cma}
  end

  def search(cma, steps) do
    case cma |> step do
      {:ok, next_cma} ->
        do_term = should_terminate(next_cma)

        if do_term do
          IO.inspect({:returning_early, cma.generation, do_term})
          {:ok, next_cma}
        else
          search(next_cma, steps - 1)
        end

      _ ->
        {:ok, cma}
    end
  end

  @spec search(CmaEs) :: {:ok, CmaEs}
  def search(cma) do
    search(cma, @max_generation)
  end

  def should_terminate(cma) do
    # NoEffectAxis: stop if adding a 0.1-standard deviation vector in any principal axis
    # direction of C does not change m
    i = cma.generation |> rem(cma.dimension)

    # m_nea = self.m + 0.1 * self.σ * tf.squeeze(self._diag_D[i] * self.B[i,:])
    m_nea =
      cma.sigma
      |> Nx.multiply(0.1)
      |> Nx.multiply(Nx.squeeze(cma.diag_D[i] |> Nx.multiply(Map.get(cma, :B)[[i, ..]])))
      |> Nx.add(cma.m)

    # m_nea_diff = tf.abs(self.m - m_nea)
    m_nea_diff = Nx.abs(cma.m |> Nx.subtract(m_nea))
    # no_effect_axis = tf.reduce_all(tf.less(m_nea_diff, self.termination_no_effect))
    no_effect_axis =
      Nx.any(Nx.less(m_nea_diff, @termination_no_effect)) |> Nx.to_flat_list() |> List.first() > 0

    # NoEffectCoord: stop if adding 0.2 stdev in any single coordinate does not change m
    # m_nec = self.m + 0.2 * self.σ * tf.linalg.diag_part(self.C)
    m_nec =
      0.2
      |> Nx.multiply(cma.sigma)
      |> Nx.multiply(Nx.take_diagonal(Map.get(cma, :C)))
      |> Nx.add(cma.m)

    # m_nec_diff = tf.abs(self.m - m_nec)
    m_nec_diff = Nx.abs(cma.m |> Nx.subtract(m_nec))
    # no_effect_coord = tf.reduce_any(tf.less(m_nec_diff, self.termination_no_effect))
    no_effect_coord =
      Nx.any(Nx.less(m_nec_diff, @termination_no_effect)) |> Nx.to_flat_list() |> List.first() > 0

    # ConditionCov: stop if the condition number of the covariance matrix becomes too large
    # max_D = tf.reduce_max(self._diag_D)
    max_D = Nx.reduce_max(cma.diag_D)
    # min_D = tf.reduce_min(self._diag_D)
    min_D = Nx.reduce_min(cma.diag_D)
    # condition_number = max_D**2 / min_D**2
    condition_number = Nx.divide(Nx.power(max_D, 2), Nx.power(min_D, 2))
    # condition_cov = tf.greater(condition_number, 1e14)
    condition_cov =
      Nx.greater(condition_number, 100_000_000_000_000) |> Nx.to_flat_list() |> List.first() > 0

    # TolXUp: stop if σ × max(D) increased by more than 10^4.
    # This usually indicates a far too small initial σ, or divergent behavior.
    # prev_max_D = tf.reduce_max(tf.linalg.diag_part(self._prev_D))
    prev_max_D = Nx.reduce_max(Nx.take_diagonal(cma.prev_D))
    # tol_x_up_diff = tf.abs(self.σ * max_D - self._prev_sigma * prev_max_D)
    tol_x_up_diff =
      Nx.abs(
        Nx.multiply(cma.sigma, max_D)
        |> Nx.subtract(Nx.multiply(cma.prev_sigma, prev_max_D))
      )

    # tol_x_up = tf.greater(tol_x_up_diff, 1e4)
    tol_x_up = Nx.greater(tol_x_up_diff, 10000) |> Nx.to_flat_list() |> List.first() > 0

    # IO.inspect({no_effect_axis, no_effect_coord , condition_cov , tol_x_up})

    no_effect_axis or no_effect_coord or condition_cov or tol_x_up
  end

  # return self.fitness_fn(tf.stack([self.m])).numpy()[0]
  def best_fitness(cma) do
    cma.fitness_function.(cma.m |> Nx.new_axis(0))
  end
end
