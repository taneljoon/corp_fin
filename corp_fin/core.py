# -*- coding: utf-8 -*-

import sys
units = 100
unit_price = 20
var_cost = 12
fixed_cost = 1000

class Operations(object):
    def __init__(self):
        self.op = []
        
    def calc(self, units):
        self.op.append(units)
    
    def get_op(self):
        return self.op
    
    def get_latest(self):
        return self.op[-1]

class Macro(object):
    def __init__(self):
        self.unit_price = []
        self.var_cost = []
    
    def calc(self, price, cost):
        self.unit_price.append(price)
        self.var_cost.append(cost)
    
    def get_price(self):
        return self.unit_price[-1]

    def get_cost(self):
        return self.var_cost[-1]
        
class Investments(object):
    def __init__(self):
        self.inv = []
    
    def invest(self, investment):
        self.inv.append(investment)
    
    def get_inv(self):
        return self.inv
        
    def get_latest(self):
        return self.inv[-1]

class Statments(object):
    def __init__(self):
        self.cf = CF()
        self.bs = BS()
        self.pl = PL()
        self.se = SE()
    
    def calc(self, op, inv, macro):
        #self.cf.calc(op, inv, macro)
        self.bs.calc(op, inv, macro)
        #self.pl.calc(op, inv, macro)
        #self.se.calc(op, inv, macro)
     
    def get_latest(self):
        return self.bs

class BS(object):
    def __init__(self):
        self.assets =[]
        self.liabilities =[]
        self.equity =[]
    
    def calc(self, op, inv, macro):
        
        self.assets.append(macro.get_price() * op.get_latest())
        self.liabilities.append(macro.get_cost() * op.get_latest())
        self.equity.append(0)
        
    def get_latest(self):
        return (self.assets, self.liabilities, self.equity)
        
class CF(object):
    def __init__(self):
        self.net = []
        
    #def calc(self, op, inv, macro):
    #    self.net.append()

class PL(object):
    def __init__(self):
        self.profit = []
        
    #def calc(

class SE(object):
    def __init__(self):
        self.retained = []

inv = Investments()    
macro = Macro()
op = Operations()
stat = Statments()

macro.calc(unit_price, var_cost)
op.calc(10)
stat.calc(op, inv, macro)


print(stat.bs.assets)

print(op.get_op())