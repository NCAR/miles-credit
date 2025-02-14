from credit.Quick_Climate_Year import run_year_rmse

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def main():
    try:
        num = int(input("Enter a number to calculate its factorial: "))
        if num < 0:
            print("Factorial is not defined for negative numbers.")
        else:
            result = factorial(num)
            print(f"The factorial of {num} is {result}.")
    except ValueError:
        print("Please enter a valid integer.")

if __name__ == "__main__":
    main()
