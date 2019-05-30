# main run file


# import libraries
import pandas as pd
import numpy as np
import datetime
import weakref
import calendar
import copy

# all classes that are timeline driven give pandas as out

class Macro(object):
    def __init__(self, init_price):
        self.price = init_price
        self.cost = init_price * 0.8
        titles = ['Datetime', 'Price', 'Cost']
        self.df = pd.DataFrame(columns = titles)

    def calc(self, timeline, delta, rate = 0):
        self.price = self.price * (1 + rate/(365.25 / delta.days))
        self.cost = self.cost * (1 + rate/(365.25 / delta.days))
        temp = {}
        temp['Datetime'] = timeline
        temp['Price'] = self.price
        temp['Cost'] = self.cost
        self.df = self.df.append(temp, ignore_index = True)
        return temp
        
class Business(object):

    instances = []
    def __init__(self, name, capacity, timeline):
        self.__class__.instances.append(weakref.proxy(self))
        self.name = name
        self.capacity = capacity
        self.op = Operations()
        self.fin = Finance(timeline)
        self.is_copy = False
        
    def get_free_cash_balance(self):
        
        return self.fin.bs.df['Assets'].values[-1]

class Operations(object):
    def __init__(self):
        titles = ['Datetime','Units']
        self.df = pd.DataFrame(columns = titles)
        
    def calc(self, timeline, delta, business):
        temp = {}
        
        units = business.capacity * delta.days
        
        temp['Datetime'] = timeline
        temp['Units'] = units
        self.df = self.df.append(temp, ignore_index = True)
        return temp



class Finance(object):
    def __init__(self, timeline):
        self.bs = BS(timeline)
        self.se = SE(timeline)
        self.cf = CF()
        self.pl = PL()
    def calc(self, timeline, revenue, cost, dividends):
        self.bs.calc(timeline, revenue, cost, dividends)
        temp = self.pl.calc(timeline, revenue, cost)
        self.cf.calc(timeline, revenue, cost, dividends)
        self.se.calc(timeline, dividends, temp['Profit'])
        
class BS(object):
    def __init__(self, timeline, assets = 0, liabilities = 0):
        titles = ['Datetime', 'Assets', 'Liabilities','Equity']
        self.df = pd.DataFrame(columns = titles)
        temp = {}
        
        temp['Datetime'] = timeline
        temp['Assets'] = assets
        temp['Liabilities'] = liabilities
        temp['Equity'] = assets - liabilities
        self.df = self.df.append(temp, ignore_index = True)

    def calc(self, timeline, revenue, cost, dividends):
        df_previous = self.df.shape[0]-1
        
        temp = {}
        temp['Datetime'] = timeline
        temp['Assets'] = self.df['Assets'][df_previous] + revenue - cost - dividends
        temp['Liabilities'] = self.df['Liabilities'][df_previous]
        temp['Equity'] = temp['Assets'] - temp['Liabilities']
        self.df = self.df.append(temp, ignore_index = True)
        return temp
        
class PL(object):
    def __init__(self):
        titles = ['Datetime','Revenues', 'Expenses', 'Profit']
        self.titles = titles
        self.df = pd.DataFrame(columns = titles)

    def calc(self, timeline, revenue, cost):
        temp = {}
        temp['Datetime'] = timeline
        temp['Revenues'] = revenue
        temp['Expenses'] = -cost
        temp['Profit'] = revenue - cost
        self.df = self.df.append(temp, ignore_index = True)
        return temp

class CF(object):
    def __init__(self):
        titles = ['Datetime','Operations','Dividends','Cash_BoP', 'Cash_EoP', 'Net_CF']
        self.titles = titles
        self.df = pd.DataFrame(columns = titles)

    def calc(self, timeline, revenue, cost, dividends, cash_bop =0):
        df_previous = self.df.shape[0]-1
        temp = {}
        try:
            temp_cash_bop = self.df['Cash_EoP'][df_previous]
        except:
            temp_cash_bop = cash_bop

        temp['Datetime'] = timeline
        temp['Operations'] = revenue - cost
        temp['Dividends'] = -dividends
        temp['Cash_BoP'] = temp_cash_bop
        temp['Net_CF'] = revenue - cost - dividends
        temp['Cash_EoP'] = temp_cash_bop + revenue - cost - dividends
        self.df = self.df.append(temp, ignore_index = True)
        return temp

class SE(object):
    def __init__(self, timeline):
        titles = ['Datetime','Balance_BoP','Profit','Dividends','Balance_EoP']
        self.titles = titles
        self.df = pd.DataFrame(columns = titles)

    def calc(self, timeline, profit, dividends, balance_bop = 0):
        df_previous = self.df.shape[0]-1
        temp = {}
        try:
            temp_balance = self.df['Balance_EoP'][df_previous]
        except:
            temp_balance = balance_bop
            
        temp['Datetime'] = timeline
        temp['Balance_BoP'] = temp_balance
        temp['Profit'] = profit
        temp['Dividends'] = -dividends
        temp['Balance_EoP'] = temp_balance + profit - dividends
        self.df = self.df.append(temp, ignore_index = True)
        return temp


class Simulation(object):
    def __init__(self, time_step_opt ='month'):
        self.time_step_opt = time_step_opt

    def time_step_datetime(self, timeline):
        """
        Return time_step in datetime
        """
        if self.time_step_opt == 'year':
            delta = timeline.replace(timeline.year + 1,12,31) - timeline
        elif self.time_step_opt == 'month':
            # take end of month day
            year = timeline.year + timeline.month // 12
            month = timeline.month % 12 + 1
            day = calendar.monthrange(year,month)[1]
            delta = timeline.replace(year,month,day) - timeline
        elif self.time_step_opt == 'day':
            delta = datetime.timedelta(days=1)
        elif self.time_step_opt == 'hour':
            delta = datetime.timedelta(seconds=60*60)
        return delta
 
class GlobalSim(Simulation):
    def __init__(self, timeline, macro, time_step_opt = 'month'):
        self.time_step_opt = time_step_opt
        self.macro = macro
        self.timeline = timeline
    def prog_time(self):
        delta = self.time_step_datetime(self.timeline)
        self.timeline = self.timeline + delta
        macro_results = macro.calc(self.timeline, delta)
        
        for business in Business.instances:
            
            self.calc(business, self.timeline, delta, macro_results)            
            
    def calc(self, business, timeline, delta, macro_results):
        op_results = business.op.calc(timeline, delta, business)
        revenue = op_results['Units'] * macro_results['Price']
        cost = op_results['Units'] * macro_results['Cost']
        dividends = business.get_free_cash_balance()
        business.fin.calc(timeline, revenue, cost, dividends)

class LocalSim(Simulation):        
    def single_calc(self, business, timeline, macro, years = 5):
        
        end_period = timeline + datetime.timedelta(years*365.25)
        
        while timeline < end_period:
            delta = self.time_step_datetime(timeline)
            timeline = timeline + delta
            macro_results = macro.calc(timeline, delta)
        
            op_results = business.op.calc(timeline, delta, business)
            revenue = op_results['Units'] * macro_results['Price']
            cost = op_results['Units'] * macro_results['Cost']
            dividends = business.get_free_cash_balance()
            business.fin.calc(timeline, revenue, cost, dividends)
        
class Valuation(object):
    def __init__(self):
        print('')
        
    def DCF_EV(self, business, current_timeline, takeover_date, discount_rate):
        business_copy = copy.deepcopy(business)
        macro_copy = copy.deepcopy(macro)
        
        local_sim = LocalSim()
        local_sim.single_calc(business_copy, current_timeline, macro_copy)
        # take away already previous periods since we are at current time
        business_copy.fin.cf.df = business_copy.fin.cf.df[business_copy.fin.cf.df['Datetime']>takeover_date]
        
        net_cf = business_copy.fin.cf.df['Net_CF'].values
        dividends = business_copy.fin.cf.df['Dividends'].values
       
        FCFt = -dividends[-1] + net_cf[-1]
        terminal_value = max(0, FCFt / discount_rate)
        net_cf[-1] = net_cf[-1] + terminal_value
        
        EV = round(np.npv(discount_rate, net_cf) - np.npv(discount_rate, dividends),0)
        
        return EV
    
    def PE_EV(self, business, multiplier):
        
        try:
            profit = business.fin.pl.df['Profit'].values[-1]
            time_step = business.fin.bs.df['Datetime'].values
        
            time_step_days = float(time_step[-1] - time_step[-2])/1e9/(60*60*24)
            EV = profit * multiplier * 365.25/time_step_days
        except:
            EV = None
        
        return EV
        

timeline = datetime.datetime(2019,1,1)
project1 = Business('Buiseness_1', 100, timeline)
project2 = Business('Buiseness_2', 200, timeline)
#sim = Simulation()
macro = Macro(10)

value = Valuation()

sim = GlobalSim(timeline, macro)
sim.prog_time()
sim.prog_time()
sim.prog_time()

DCF_EV = value.DCF_EV(project2, sim.timeline, sim.timeline + datetime.timedelta(60), 0.1/12)
PE_EV = value.PE_EV(project2, 10)
print(DCF_EV)
print(PE_EV)

print(project2.fin.pl.df)
print(project2.fin.cf.df)
print(project2.op.df)
print(timeline)

"""
for ii in range(0,10):

    # get timestep
    delta = sim.time_step_datetime(timeline)
    timeline = timeline + delta
    macro_results = macro.calc(timeline, delta)
    
    for business in Business.instances:
        
        op_results = business.op.calc(timeline, delta, business)
        
        revenue = op_results['Units'] * macro_results['Price']
        cost = op_results['Units'] * macro_results['Cost']
        
        dividends = business.get_free_cash_balance()
        business.fin.calc(timeline, revenue, cost, dividends)
      
        
print(macro.df)
print(project1.op.df)
print(project1.fin.bs.df)
print(project1.fin.pl.df)
print(project1.fin.se.df)
print(project1.fin.cf.df)
print(project1.op.df)
"""