"""Script de teste que utiliza apenas o módulo compilado `fob_func`"""
from fob_func import fob
def main():
    exemplos = [
        [1, 1],
        [420.9687, 420.9687],
        [-100, 200],
        [-9779779779.779778,  9779779779.779778]
    ]
    for X in exemplos:
        print(f"fob({X}) = {fob(X):.6f}")


if __name__ == "__main__":
    main()
