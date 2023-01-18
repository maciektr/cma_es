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
end
