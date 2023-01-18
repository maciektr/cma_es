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

    m = cma.m |> Nx.to_flat_list() |> Enum.at(0)
    assert m < 0.001
    assert m > -0.001

    best_fn = CmaEs.best_fitness(cma) |> Nx.to_flat_list() |> Enum.at(0)
    assert best_fn < 0.001
    assert best_fn > -0.001
  end

  test "test six_hump_camel function" do
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
    end

    cma =
      %CmaEs{
        fitness_function: fitness_fn,
        initial_solution: Nx.tensor([1.5, 2.4]),
        initial_step_size: 0.5
      }
      |> CmaEs.init()

    {:ok, cma} = CmaEs.search(cma)

    [x1, x2] = cma.m |> Nx.to_flat_list()

    best_fn = CmaEs.best_fitness(cma) |> Nx.to_flat_list() |> Enum.at(0)

    # IO.inspect({x1, x2, best_fn})

    assert (x1 < 0.0898 + 0.001 and x1 > 0.0898 - 0.001 and x2 < -0.7126 + 0.001 and
              x2 > -0.7126 - 0.001) or
             (x1 < -0.0898 + 0.001 and x1 > -0.0898 - 0.001 and x2 < 0.7126 + 0.001 and
                x2 > 0.7126 - 0.001)

    assert best_fn < -1.0316 + 0.001 and best_fn > -1.0316 - 0.001
  end

  test "test branin function" do
    pi = 3.141592653589793

    fitness_fn = fn x ->
      a = 1.0
      b = 5.1 / (4 * pi ** 2)
      c = 5.0 / pi
      r = 6.0
      s = 10.0
      t = 1 / (8 * pi)

      a
      |> Nx.multiply(
        x[[.., 1]]
        |> Nx.subtract(b |> Nx.multiply(x[[.., 0]] |> Nx.power(2)))
        |> Nx.add(c |> Nx.multiply(x[[.., 0]]))
        |> Nx.subtract(r)
      )
      |> Nx.power(2)
      |> Nx.add(s |> Nx.multiply(1 |> Nx.subtract(t)) |> Nx.multiply(Nx.cos(x[[.., 0]])))
      |> Nx.add(s)
    end

    cma =
      %CmaEs{
        fitness_function: fitness_fn,
        initial_solution: Nx.tensor([-2.0, 7.0]),
        initial_step_size: 1.0
      }
      |> CmaEs.init()

    {:ok, cma} = CmaEs.search(cma)

    [x1, x2] = cma.m |> Nx.to_flat_list()

    best_fn = CmaEs.best_fitness(cma) |> Nx.to_flat_list() |> Enum.at(0)

    # IO.inspect({x1, x2, best_fn})

    a = -pi
    b = 12.275
    c = pi
    d = 2.275
    e = 9.42478
    f = 2.475
    r = 0.01

    assert (x1 < a + r and x1 > a - r and x2 < b + r and x2 > b - r) or
             (x1 < c + r and x1 > c - r and x2 < d + r and x2 > d - r) or
             (x1 < e + r and x1 > e - r and x2 < f + r and x2 > f - r)
  end
end
