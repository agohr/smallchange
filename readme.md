# smallchange

## Overview

*smallchange* is a program that *efficiently* solves the change counting problem. In other words, given an amount of currency units *N* and a list of denominations *D = [d_1, d_2, d_3,..., d_M]*, it will compute and output the number of ways *C(N,D)* to give change for *N* currency units in the given denominations, not counting solutions that only differ by order of coins.

In principle, this is a well-known, easy dynamic programming problem, and there are many programs freely available that solve it. However, these standard solutions suffer from two grave problems:

1. They have a runtime and memory complexity of *O(NM)*. This is fine for small instances, but becomes problematic if the user is, say, a national government trying to figure out the number of ways to give change for their budget.

2. They do not give a human-verifiable proof of the results they print out.

*smallchange* aims to improve upon these solutions on both *counts*.

## Main Idea

The basic mathematical idea behind *smallchange* is simple: one can prove by induction over the number of denominations that the number of ways to give change for *N* currency units using a given set of denominations *D = [d_1, d_2, d_3, ..., d_M]* can be expressed as a polynomial with rational coefficients of degree at most *M-1*, as long as the residue class of *n mod lcm(D)* is known in advance. *smallchange* calculates these polynomials explicitly (by interpolation from suitably chosen support points) and uses them to evaluate *C(N,D)*. In the process, it opportunistically uses the standard recurrence relations on *C(N,D)* when this is simpler. 

## Installation

Clone this repository and navigate to the smallchange directory. Make sure that you have Python3 installed. No other requirements should be necessary.

## Usage

Run the `smallchange_main.py` script with the following command:

```
python smallchange_main.py --amount AMOUNT --denominations DENOMINATIONS_CSV --verbose
```

- `AMOUNT`: The amount of money for which you want to calculate the number of ways to give change (default: 100).
- `DENOMINATIONS_CSV`: A CSV file containing the coin denominations (default: British.csv).
- `--verbose`: (Optional) Output the main steps of the computation.

Example:

```
python smallchange_main.py --amount 500 --denominations British.csv --verbose
```

## Example Output

```
> python smallchange_main.py --amount 1000000000000
Number of ways to make 1000000000000 with denominations [1, 2, 5, 10, 20, 50, 100]: 138888888967222222238381944445906111111163675000000365000000001
```

## Testing

To run the tests, execute the `change_coins.py` script:

```
python change_coins.py
```

This will run a series of tests to validate the correctness of the implemented functions.

## License

This project is licensed under the MIT License.

## Acknowledgments

Parts of *smallchange* were generated by GPT-4 upon instruction by the author, namely the matrix code, the polynomial recovery function, the tests, and the user facing logic.