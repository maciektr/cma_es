defmodule CmaEsTest do
  use ExUnit.Case
  doctest CmaEs

  test "test one dim" do
    fitness_fn = fn x -> x[[.., 0]] |> Nx.power(2) end

    cma =
      %CmaEs{
        fitness_function: fitness_fn,
        initial_solution: Nx.tensor([3.0]),
        initial_step_size: 1.0
      }
      |> CmaEs.init()

    {:ok, cma} = CmaEs.search(cma)

    # IO.inspect({:CMA, cma})

    m = cma.m |> Nx.to_flat_list() |> Enum.at(0)
    assert m < 0.001
    assert m > -0.001

    best_fn = CmaEs.best_fitness(cma) |> Nx.to_flat_list() |> Enum.at(0)
    assert best_fn < 0.001
    assert best_fn > -0.001
  end

  test "test six_hump_camel" do
    # Six-Hump Camel Function
    # https://www.sfu.ca/~ssurjano/camel6.html

    fitness_fn = fn x ->
      4
      |> Nx.subtract(2.1 |> Nx.multiply(x[[.., 0]] |> Nx.power(2)))
      |> Nx.add(x[[.., 0]] |> Nx.power(4) |> Nx.divide(3))
      |> Nx.multiply(x[[.., 0]] |> Nx.power(2))
      |> Nx.add(x[[.., 0]] |> Nx.multiply(x[[.., 1]]))
      |> Nx.add(
        -4
        |> Nx.add(4 |> Nx.multiply(x[[.., 1]] |> Nx.power(2)))
        |> Nx.multiply(x[[.., 1]] |> Nx.power(2))
      )

      # 4
      # |> Nx.subtract(2.1 |> Nx.dot(x[[.., 0]] |> Nx.power(2)))
      # |> Nx.add(x[[.., 0]] |> Nx.power(4) |> Nx.divide(3))
      # |> Nx.dot(x[[.., 0]] |> Nx.power(2))
      # |> Nx.add(x[[.., 0]] |> Nx.dot(x[[.., 1]]))
      # |> Nx.add(
      #   -4
      #   |> Nx.add(4 |> Nx.dot(x[[.., 1]] |> Nx.power(2)))
      #   |> Nx.multiply(x[[.., 1]] |> Nx.power(2))
      # )
    end

    cma =
      %CmaEs{
        fitness_function: fitness_fn,
        initial_solution: Nx.tensor([1.5, 2.4]),
        initial_step_size: 0.5
      }
      |> CmaEs.init()

    {:ok, cma} = CmaEs.search(cma)

    # m = cma.m |> Nx.to_flat_list() |> Enum.at(0)
    [x1, x2] = cma.m |> Nx.to_flat_list()
    # IO.inspect({:m, m})

    best_fn = CmaEs.best_fitness(cma) |> Nx.to_flat_list() |> Enum.at(0)

    IO.inspect({x1, x2, best_fn})

    assert (x1 < 0.0898 + 0.001 and x1 > 0.0898 - 0.001 and x2 < -0.7126 + 0.001 and
              x2 > -0.7126 - 0.001) or
             (x1 < -0.0898 + 0.001 and x1 > -0.0898 - 0.001 and x2 < 0.7126 + 0.001 and
                x2 > 0.7126 - 0.001)

    assert best_fn < -1.0316 + 0.001 and best_fn > -1.0316 - 0.001

    # Assert global minimum has been reached
    # cond = (
    #     (
    #         np.isclose(x1, 0.0898, rtol=1e-3) and
    #         np.isclose(x2, -0.7126, rtol=1e-3)
    #     ) or
    #     (
    #         np.isclose(x1, -0.0898, rtol=1e-3) and
    #         np.isclose(x2, 0.7126, rtol=1e-3)
    #     )
    # )
    # self.assertTrue(cond)

    # self.assertTrue(np.isclose(best_fitness, -1.0316, rtol=1e-3))

    # # Early stopping occured
    # self.assertTrue(cma.generation < num_max_epochs)
  end

  # def test_six_hump_camel_fn(self):
  #     num_max_epochs = 100

  #     def fitness_fn(x):
  #         """
  #         Six-Hump Camel Function
  #         https://www.sfu.ca/~ssurjano/camel6.html
  #         """
  # return (
  #     (4 - 2.1 * x[:,0]**2 + x[:,0]**4 / 3) * x[:,0]**2 +
  #     x[:,0] * x[:,1] +
  #     (-4 + 4 * x[:,1]**2) * x[:,1]**2
  # )

  #     cma = CMA(
  #         initial_solution=[1.5, 2.4],
  #         initial_step_size=0.5,
  #         fitness_function=fitness_fn
  #     )

  #     (x1, x2), best_fitness = cma.search(num_max_epochs)

  #     # Assert global minimum has been reached
  #     cond = (
  #         (
  #             np.isclose(x1, 0.0898, rtol=1e-3) and
  #             np.isclose(x2, -0.7126, rtol=1e-3)
  #         ) or
  #         (
  #             np.isclose(x1, -0.0898, rtol=1e-3) and
  #             np.isclose(x2, 0.7126, rtol=1e-3)
  #         )
  #     )
  #     self.assertTrue(cond)

  #     self.assertTrue(np.isclose(best_fitness, -1.0316, rtol=1e-3))

  #     # Early stopping occured
  #     self.assertTrue(cma.generation < num_max_epochs)
end
