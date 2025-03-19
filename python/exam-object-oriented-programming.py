# Exercise: Warm-up

# (a) Define a Drink class which contains 2 attributes: volume and expiration. The volume attribute corresponds to the volume in milliliters of the container containing the drink and expiration corresponds to the number of days remaining before the drink expires. The values of these two attributes must be passed as arguments to the constructor of the Drink class.

# (b) Define in the Drink class a next_day method which will reduce by 1 the number of days before the drink's peremption.

# (c) Instantiate a first Drink object whose volume will be 350mL and whose expiration date is in 2 weeks.

class Drink:
    def __init__(self, volume, expiration):
        self.volume = volume
        self.expiration = expiration
        
    def next_day(self):
        if self.expiration > 0:
            self.expiration -= 1
            print('The expiration date was updated (-1)')
        else:
            print('The drink has already expired! :-(')

first_drink = Drink(350, 14)

first_drink.next_day()

# The expiration date was updated (-1)
# (d) Define a Juice class which will inherit from the Drink class. It will be assumed that fruit juices systematically expire after a week.

# (e) Define a DataCola class which will inherit from the Drink class. The constructor of this class will have a single argument which corresponds to the drink's container. If the drink comes in a 'can', it will automatically have a volume of 330mL and can be stored for 60 days. If the drink comes in a 'bottle', its volume will be 500mL and can be stored for 30 days. You do not need to add attributes to the DataCola class.

# (f) Instantiate an object of type Juice whose volume will be 1000mL and an object of type DataCola whith a can container.

class Juice(Drink):
    def __init__(self, volume):
        self.volume = volume
        self.expiration = 7
        
class DataCola(Drink):
    def __init__(self, container):
        if container == 'can':
            self.volume = 330
            self.expiration = 60
        elif container == 'bottle':
            self.volume = 500
            self.expiration = 30
        else:
            raise ValueError('Container must be either can or bottle!')
            
suco = Juice(1000)

coca = DataCola('can')

# (g) Define a new VendingMachine class which will have two attributes:content which will be a list of objects of type Drink and size which corresponds to the number of drinks that the vending machine can contain at most.

# (h) Define in the VendingMachine class an add method which will take as argument an object of type Drink and will add it at the end of the list of drinks contained in a vending machine if there is still room in it.

# (i) Define in the class VendingMachine a method remove which will take as argument an integer i and will remove the i-th element of the list of drinks contained in the vending machine. You can use the pop(i) method of the list class which removes the i-th element from the list calling the method.

# (j) Define in the VendingMachine class a verify method which should check that will check that none of the drink is out of date. If one of these drinks have expired, it must be removed from the vending machine.

# (k) Define in the VendingMachine class anext_day method which will reduce the number of days of conservation left for all the drinks in the dispenser (Vending Machine).

class VendingMachine:
    def __init__(self, size):
        self.content = []
        self.size = size
        
    def add_drink(self, drink):
        if isinstance(drink, Drink):
            if len(self.content) < self.size:
                self.content.append(drink)
                print('Drink is added!')
            else:
                print('Vending Machine is full. Drink not added!')
        else:
            print('You can only use drinks of class Drink!')
    
    def remove_drink(self, i):
        if len(self.content) < i:
            print('The list has only', len(self.content), 'items!')
        else:
            self.content.pop(i)
            print('The', i, 'th item was removed!')
            
    def verify(self):
        for i, drink in enumerate(self.content):
            if (drink.expiration <= 0):
                self.remove_drink(i)
            else:
                print(i, 'th Drink is still good!')
                
    def next_day(self):
        for j, drink in enumerate(self.content):
            if (drink.expiration > 0):
                drink.expiration -= 1
                print('Expiration date on item', j, 'was updated!')
            else:
                drink.pop(j)
                print('The drink (', j, ') has already expired and was removed!')
                
# Tests        
vending_machine = VendingMachine(3)

cola_can = Drink(330, 3)
juice = Drink(250, 2)

print(vending_machine.add_drink(cola_can))
print(vending_machine.add_drink(juice))

print(vending_machine.next_day())
print(vending_machine.verify())
print(vending_machine.remove_drink(0))

# Exercise: Tic-Tac-Toe

# (a) Define a Cell class which contains a single attribute named occupied. This attribute will automatically take the value ' ' (blank space) during instantiation.

# (b) Define in the class Cell a method play1 which will give the value 'X' to the attribute occupied if the cell is not already occupied.

# (c) Define in the class Cell a method play2 which will give the value 'O' to the attribute occupied if the cell is not already occupied.

class Cell:
    def __init__(self):
        self.occupied = ' '
        # I added this, since due to the occupied status it doesn't keep that
        # X plays, if s/he chose a cell that is occupied
        self.error = False
        
    def play1(self):
        if self.occupied == ' ':
            self.occupied = 'X'
        else:
            self.error = True
            print('Cell is already occupied!')

    def play2(self):
        if self.occupied == ' ':
            self.occupied = 'O'
        else:
            self.error = True
            print('Cell is already occupied!')
            
    def __str__(self):
        return self.occupied

# (d) Define a Board class which has two attributes: grid and turn. The grid attribute is a 9-element list of objects of type Cell. The turn attribute is an integer which is equal to 1 if it is the turn of player 1 to play and 2 if it is the turn of player 2. The turn attribute will be automatically initialized with the value 1.

# (e) Define in the Board class the __str__ method which will allow you to use the print function on objects of this class. The print function should display in a first line the content of cells 0 to 2, then in a second line the content of cells 3 to 5 and finally in a third line the content of cells 6 to 8. The cells will be separated by the character '|' and each line will end with the character '\n' which is the end-of-line character.

# (f) Define in the Board class a play method which will take as a parameter an integer ranging from 0 to 8. Depending on the player whose turn it is to play, this method will call the play1 or play2 methods on the cell corresponding to the integer passed as argument. It will then be necessary to modify the value of the turn attribute so that the next player can play.

class Board:
    def __init__(self):
        self.grid = [Cell() for _ in range(9)]
        self.turn = 1
        
    def __str__(self):
        l1 = str(self.grid[0]) + '|' + str(self.grid[1]) + '|' + str(self.grid[2])
        l2 = str(self.grid[3]) + '|' + str(self.grid[4]) + '|' + str(self.grid[5])
        l3 = str(self.grid[6]) + '|' + str(self.grid[7]) + '|' + str(self.grid[8])

        return l1 + '\n' + l2 + '\n' + l3

    def play(self, field: int):
        assert field in range(0,8), 'The number input is out of range! Range: [0,8]'
        if (self.turn % 2 == 0):
            self.grid[field].play2()
            # I added the print, so it is better to play
            print(self)
            #  added instructions
            self.endofgame(self.grid[field])
            # only if field is not occupied ownership of turn changes!
            if self.grid[field].error == False:
                self.turn += 1
        else:
            self.grid[field].play1()
            # I added the print, so it is better to play
            print(self)
            #  added instructions
            self.endofgame(self.grid[field])
            # only if field is not occupied ownership of turn changes!
            if self.grid[field].error == False:
                self.turn += 1

    def endofgame(self, player):
        # Test if first row is full
        if (str(self.grid[0]) == str(self.grid[1]) == str(self.grid[2])) & (str(self.grid[2]) != ' '):
            winner = self.grid[0]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if second row is full
        elif (str(self.grid[3]) == str(self.grid[4]) == str(self.grid[5])) & (str(self.grid[5]) != ' '):
            winner = self.grid[3]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if third row is full
        elif (str(self.grid[6]) == str(self.grid[7]) == str(self.grid[8])) & (str(self.grid[8]) != ' '):
            winner = self.grid[6]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if first column is full
        elif (str(self.grid[0]) == str(self.grid[3]) == str(self.grid[6])) & (str(self.grid[6]) != ' '):
            winner = self.grid[0]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if second column is full
        elif (str(self.grid[1]) == str(self.grid[4]) == str(self.grid[7])) & (str(self.grid[7]) != ' '):
            winner = self.grid[1]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if third column is full
        elif (str(self.grid[2]) == str(self.grid[5]) == str(self.grid[8])) & (str(self.grid[8]) != ' '):
            winner = self.grid[2]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if first diagonal is full
        elif (str(self.grid[0]) == str(self.grid[4]) == str(self.grid[8])) & (str(self.grid[8]) != ' '):
            winner = self.grid[1]
            print('End of game!', winner, 'has won! Congrats!')
        # Test if second diagonal is full
        elif (str(self.grid[2]) == str(self.grid[4]) == str(self.grid[6])) & (str(self.grid[6]) != ' '):
            winner = self.grid[2]
            print('End of game!', winner, 'has won! Congrats!')
        else: 
            if (str(player) == 'X'):
                print('Next turn from : O')
            else:
                print('Next turn from: X')

# (g) Write a series of instructions to obtain the following display:
#    |   | O 
#  X | X | X 
#  O |   |   

board = Board()

board.play(3)
#  | | 
# X| | 
#  | | 
# Next turn from : O

board.play(2)
#  | |O
# X| | 
#  | | 
# Next turn from: X

board.play(4)
#  | |O
# X|X| 
#  | | 
# Next turn from : O

board.play(6)
#  | |O
# X|X| 
# O| | 
# Next turn from: X

board.play(5)
#  | |O
# X|X|X
# O| | 
# End of game! X has won! Congrats!