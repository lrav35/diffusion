import os
import shutil

def spacers(): return "\n" + "#" * shutil.get_terminal_size().columns + "\n" 

def main():
    print(spacers())
    print("Would you like to begin the application? ('quit' + <Enter> to exit)\n")

    while 1:
        usr = input('>>>')
        if usr == "quit": 
            print("\nprogram ending... (╥ꞈ╥)", "red")
            break
        elif usr == '':
            print("continue to next image")
            # continue to next image
        elif usr == 'n':
            print("next")
            # iterate to next index in mask until good one is found
        else:
            print('inuput not recognize.. try again')

if __name__ == '__main__':
    main()