"Создание класса Birds с атрибутами (имя, способность летать, размер),"
" методами (пение, еда, сон, дальность полёта) и деструктором."
" Создание объекта воробей и вызов метода flying_range()"

class Birds:

    def __init__(self,name,flying_ability,size):
        self.name = name
        self.flying_ability = flying_ability
        self.size = size
        
    def soinding (self):
        print(f"птичка {self.name} поёт утром")

    def eating (self):
        print(f"птичка {self.name} ест днём")

    def sleeping (self):
        print(f"птичка {self.name} спит ночью")

    def flying_range(self,N):
        if self.flying_ability ==True:
            print( self.size * N)

        else:
            print ("error ")
 
    def __del__(self):
        print("del done")

vorobey  = Birds ("vorobey",True,50 )      
vorobey.flying_range (10)
print(" привет")