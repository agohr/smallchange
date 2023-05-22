import argparse
import csv

import change_coins as cc

def main():
    parser = argparse.ArgumentParser(description='Efficiently solve the change counting problem.')
    parser.add_argument('--amount', type=int, default=100, help='Amount of money N (default: 100)')
    parser.add_argument('--denominations', type=str, default='British.csv', help='CSV file with coin denominations (default: British.csv)')
    parser.add_argument('--verbose', action='store_true', help='Output the main steps of the computation')

    args = parser.parse_args()

    with open(args.denominations, 'r') as csvfile:
        reader = csv.reader(csvfile)
        denominations = [int(row) for row in next(reader)]

    if args.verbose:
        print(f'Amount: {args.amount}')
        print(f'Denominations: {denominations}')

    n = cc.change_number_rec(args.amount, denominations, verbose=args.verbose)
    print(f'Number of ways to make {args.amount} with denominations {denominations}: {n}')

if __name__ == '__main__':
    main()