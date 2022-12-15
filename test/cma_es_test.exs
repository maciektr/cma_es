defmodule CmaEsTest do
  use ExUnit.Case
  doctest CmaEs

  test "test one dim" do
    fitness_fn = fn x -> x[[.., 0]] ** 2 end

    cma =
      %CmaEs{
        fitness_function: fitness_fn,
        initial_solution: Nx.tensor([3.0]),
        initial_step_size: 1.0
      }
      |> CmaEs.init()

    cma = CmaEs.search(cma)

    cma |> IO.puts()
  end
end
