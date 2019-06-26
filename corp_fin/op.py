# main run file


"""
# notes

# corp fin is kinda crappy - equity rasing or starting subsidiaries doesnt work
"""

# import libraries
import pandas as pd
import numpy as np
import datetime
import weakref
import calendar
import copy

# import libraries from disc
import sys
sys.path.insert(0,'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/for_import')

import folium
#sys.exit()

# all classes that are timeline driven give pandas as out

class Macro(object):
    def __init__(self, init_price):
        self.price = init_price
        self.cost = init_price * 0.8
        self.capex = init_price * 365
        titles = ['Datetime', 'Price', 'Cost', 'CAPEX']
        self.df = pd.DataFrame(columns = titles)

    def calc(self, timeline, delta, rate = 0):
        self.price = self.price * (1 + rate/(365.25 / delta.days))
        self.cost = self.cost * (1 + rate/(365.25 / delta.days))
        self.capex = self.capex * (1 + rate/(365.25 / delta.days))
        temp = {}
        temp['Datetime'] = timeline
        temp['Price'] = self.price
        temp['Cost'] = self.cost
        temp['CAPEX'] = self.capex
        self.df = self.df.append(temp, ignore_index = True)
        return temp
        
class Business(object):

    instances = []
    def __init__(self, name, capacity, timeline, current_assets=0, 
        fixed_assets=0, liabilities=0, investments=0, estimate = False):
        
        if estimate == False:
            self.__class__.instances.append(weakref.proxy(self))
        self.name = name
        self.type = 'Business'
        self.capacity = capacity
        self.op = Operations()
        self.fin = Finance(timeline, current_assets, fixed_assets, liabilities, investments)
        
    def get_dividends(self):
        try:
            df_previous = self.fin.bs.df.shape[0]-1
            temp_balance = self.fin.bs.df['Current_assets'][df_previous]
            if self.fin.investments !=0:
                temp_balance = 0
        except:
            temp_balance = 0
        
        return max(temp_balance,0)
        
    def get_business_investments(self):
        return 0

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
    def __init__(self, timeline, current_assets=0, fixed_assets=0, 
        liabilities=0, investments=0):
        self.bs = BS(timeline, current_assets, fixed_assets, liabilities, investments)
        self.se = SE(timeline, current_assets, fixed_assets, liabilities, investments)
        self.cf = CF(investments)
        self.pl = PL()
        # investment goes like this - cash to current assets and equity
        # - then cash to fixed assets and cash flow negative
        self.investments = investments
        
    def calc(self, timeline, revenue, cost, dividends, investments):
        self.bs.calc(timeline, revenue, cost, dividends, investments, self.investments)
        temp = self.pl.calc(timeline, revenue, cost)
        self.cf.calc(timeline, revenue, cost, dividends, investments, self.investments)
        self.se.calc(timeline, dividends, temp['Profit'], investments)
        self.investments = 0
        
class BS(object):
    def __init__(self, timeline, current_assets = 0, fixed_assets=0, 
        liabilities=0, investments=0):
        titles = ['Datetime', 'Current_assets', 'Fixed_assets', 'Total_assets', 'Liabilities','Equity']
        self.df = pd.DataFrame(columns = titles)
        
        temp = {}
        temp['Datetime'] = timeline
        temp['Current_assets'] = current_assets + investments
        temp['Fixed_assets'] = fixed_assets
        temp['Total_assets'] = temp['Fixed_assets'] + temp['Current_assets'] 
        temp['Liabilities'] = liabilities
        temp['Equity'] = temp['Total_assets'] - temp['Liabilities']
        self.df = self.df.append(temp, ignore_index = True)

    def calc(self, timeline, revenue, cost, dividends, investments, current_to_fixed):
        df_previous = self.df.shape[0]-1
        
        temp = {}
        temp['Datetime'] = timeline
        
        temp['Current_assets'] = (self.df['Current_assets'][df_previous] +
        revenue - cost - dividends - current_to_fixed + investments)
        temp['Fixed_assets'] = self.df['Fixed_assets'][df_previous] + current_to_fixed
        temp['Total_assets'] = temp['Fixed_assets'] + temp['Current_assets'] 
        
        temp['Liabilities'] = self.df['Liabilities'][df_previous]
        temp['Equity'] = temp['Total_assets'] - temp['Liabilities']
        self.df = self.df.append(temp, ignore_index = True)
        return temp
        
class PL(object):
    def __init__(self):
        titles = ['Datetime','Revenues', 'Expenses', 'Profit']
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
    def __init__(self, investments=0):
        titles = (['Datetime','Cash_BoP','Operations','Equity_raised',
        'Investments','Dividends', 'Cash_EoP','Net_CF'])
        self.df = pd.DataFrame(columns = titles)
        #self.cash_init = fixed_assets - liabilities
        self.cash_init = investments

    def calc(self, timeline, revenue, cost, dividends, investments, current_to_fixed):
        df_previous = self.df.shape[0]-1
        temp = {}
        try:
            temp_cash_bop = self.df['Cash_EoP'][df_previous]
        except:
            temp_cash_bop = self.cash_init

        temp['Datetime'] = timeline
        temp['Cash_BoP'] = temp_cash_bop
        temp['Operations'] = revenue - cost
        temp['Equity_raised'] = investments
        temp['Investments'] = -current_to_fixed
        temp['Dividends'] = -dividends
        temp['Net_CF'] = revenue - cost - dividends
        temp['Cash_EoP'] = temp_cash_bop + revenue - cost - dividends - current_to_fixed
        self.df = self.df.append(temp, ignore_index = True)
        return temp

class SE(object):
    def __init__(self, timeline, current_assets=0, fixed_assets=0, 
        liabilities=0, investments=0):
        titles = ['Datetime','Balance_BoP','Equity_raised','Profit','Dividends','Balance_EoP']
        self.df = pd.DataFrame(columns = titles)
        
        temp = {}
        temp['Datetime'] = timeline
        temp['Balance_BoP'] = 0 #fixed_assets + current_assets
        temp['Equity_raised'] = investments #fixed_assets + current_assets
        temp['Profit'] = 0
        temp['Dividends'] = 0
        temp['Balance_EoP'] = investments
        self.df = self.df.append(temp, ignore_index = True)

    def calc(self, timeline, dividends, profit, equity_raised):
        
        df_previous = self.df.shape[0]-1
        temp_balance = self.df['Balance_EoP'][df_previous]
        
        temp = {}     
        temp['Datetime'] = timeline
        temp['Balance_BoP'] = temp_balance
        temp['Equity_raised'] = equity_raised
        temp['Profit'] = profit
        temp['Dividends'] = -dividends
        temp['Balance_EoP'] = temp_balance + profit - dividends + equity_raised
        self.df = self.df.append(temp, ignore_index = True)
        return temp


class Simulation(object):
    def __init__(self, time_step_opt ='month'):
        self.time_step_opt = time_step_opt

    def time_step_datetime(self, timeline):
        """
        Return time_step in datetime.
        If not end of a certain period give delta to closest end of period.
        """
        if self.time_step_opt == 'year':
            if timeline != datetime.datetime(timeline.year,12,31):
                delta = timeline.replace(timeline.year,12,31) - timeline
            else:    
               delta = timeline.replace(timeline.year + 1,12,31) - timeline
               
        elif self.time_step_opt == 'month':
            # take end of month day
            year = timeline.year
            month = timeline.month
            day = calendar.monthrange(year,month)[1]
            
            if timeline != datetime.datetime(year,month,day):
                delta = timeline.replace(year,month,day) - timeline
            else:
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
    def __init__(self, timeline, macro, events, time_step_opt = 'month'):
        self.time_step_opt = time_step_opt
        self.macro = macro
        self.timeline = timeline
        self.events = events
        
    def prog_time(self):
        """
        Main time progression.
        """
        delta = self.time_step_datetime(self.timeline)
        self.timeline = self.timeline + delta
        macro_results = self.macro.calc(self.timeline, delta)
        
        for business in Business.instances:            
            self.calc(business, self.timeline, delta, macro_results)

        for corporation in Corporation.instances:
            self.calc_corp(corporation, self.timeline, delta, macro_results)

    def calc(self, business, timeline, delta, macro_results):
        op_results = business.op.calc(timeline, delta, business)
        revenue = op_results['Units'] * macro_results['Price']
        cost = op_results['Units'] * macro_results['Cost']
        
        dividends = business.get_dividends()
        investments = business.get_business_investments()
        business.fin.calc(timeline, revenue, cost, dividends, investments)
    
    def calc_corp(self, corporation, timeline, delta, macro_results):
        
        revenue = 0
        cost = 0
        dividends = 0
        investments = 0
        
        for business in corporation.businesses:
            revenue = revenue + business.fin.pl.df['Revenues'].values[-1]
            cost = cost + business.fin.pl.df['Expenses'].values[-1]
            dividends = dividends + business.fin.cf.df['Dividends'].values[-1]
            investments = investments + business.fin.cf.df['Investments'].values[-1]
        
        temp = self.events.create_business(corporation, self.macro, 
            macro_results, self.timeline, self.time_step_opt)
        investments = temp + investments
        corporation.fin.calc(timeline, revenue, -cost, dividends, -investments)

class LocalSim(Simulation):        
    def single_calc(self, entity, timeline, macro, years = 5):
        """
        Calculate financial projection.
        """
        end_period = timeline + datetime.timedelta(years*365.25)
        
        while timeline < end_period:
            delta = self.time_step_datetime(timeline)
            timeline = timeline + delta
            macro_results = macro.calc(timeline, delta)
            
            investments = 0
            op_results = entity.op.calc(timeline, delta, entity)
            revenue = op_results['Units'] * macro_results['Price']
            cost = op_results['Units'] * macro_results['Cost']
            
            if entity.type == 'Business':
                dividends = entity.get_dividends()
            elif entity.type == 'Corporation':
                dividends = 0
            
            entity.fin.calc(timeline, revenue, cost, dividends, 0)
        
    def single_corp_calc(self):
        """
        xxx finish this corp sim
        """
        print('xxx')
    
class Valuation(object):
    def __init__(self):
        print('')
        
    def DCF_EV(self, business, macro, current_timeline, takeover_date, discount_rate):
        """
        Copy the business and call fin-projection.
        Using DCF get NPV of dividends and net cash flow.
        """
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
    
    def PE_EV(self, business, macro, multiplier, current_timeline):
        """
        Calculate EV based on last profit.
        Datetime difference gives answer in nanoseconds, therefore convert to dayss.
        """
        try:
            profit = business.fin.pl.df['Profit'].values[-1]
            time_step = business.fin.bs.df['Datetime'].values
            time_step_days = float(time_step[-1] - time_step[-2])/1e9/(60*60*24)
            EV = profit * multiplier * 365.25/time_step_days
        except:
            local_sim = LocalSim()
            
            business_copy = copy.deepcopy(business)
            macro_copy = copy.deepcopy(macro)
            
            local_sim.single_calc(business_copy, current_timeline, macro_copy, years = 1)
            
            profit = business_copy.fin.pl.df['Profit'].values[-1]
            time_step = business_copy.fin.bs.df['Datetime'].values
            time_step_days = float(time_step[-1] - time_step[-2])/1e9/(60*60*24)
            EV = profit * multiplier * 365.25/time_step_days
        
        return EV
        
    def IRR_business(self, business, macro, current_timeline, takeover_date,
        investment, discount_rate, time_step_opt):
        
        business_copy = copy.deepcopy(business)
        macro_copy = copy.deepcopy(macro)
        
        local_sim = LocalSim()
        local_sim.single_calc(business_copy, current_timeline, macro_copy)
        # take away already previous periods since we are at current time
        business_copy.fin.cf.df = business_copy.fin.cf.df[business_copy.fin.cf.df['Datetime']>takeover_date]
                
        net_cf = business_copy.fin.cf.df['Net_CF'].values
        dividends = business_copy.fin.cf.df['Dividends'].values
        
        investments = business_copy.fin.cf.df['Investments'].values
        
        FCFt = -dividends[-1] + net_cf[-1]
        terminal_value = max(0, FCFt / discount_rate)
        #net_cf[-1] = net_cf[-1] + terminal_value
        
        temp_list = investments - dividends
        temp_list[-1] = temp_list[-1] + terminal_value 
        
        
        print(business_copy.fin.bs.df)
        print(business_copy.fin.se.df)
        
        print('xxx')
        print('dividends')
        print(dividends)
        
        print('investments')
        print(investments)
        
        print('xxx')
        
        print(temp_list)
        
        IRR = round(np.irr(temp_list),5)
        if time_step_opt == 'month':
            IRR_yearly = IRR *12
        elif time_step_opt == 'day':
            IRR_yearly = IRR * 365
        else:
            IRR_yearly = IRR

        return round(IRR_yearly,5)

class Corporation(object):
    instances = []
    def __init__(self, name, timeline, businesses=[], current_assets=0, 
        fixed_assets=0, liabilities=0, human = False, estimate = False):
        
        if estimate == False:
            self.__class__.instances.append(weakref.proxy(self))
        self.name = name
        self.type = "Corporation"
        self.businesses = businesses
        self.fin = Finance(timeline, current_assets, fixed_assets, liabilities)                 
        self.human = human
        
    def get_dividends(self):
        return self.fin.bs.df['Current_assets'].values[-1]
    
    def get_investments(self):
        # xxx 
        return 0
    
    def append_business(self, name, capacity, timeline, capex):
        """
        Appends new business to the corporation.
        """
        self.businesses.append(Business(name, capacity, timeline, investments=capex ))
        
    def buy_business(self, business):
        """
        xxx
        """
        return 0

class Events(object):
    def __init__(self):
        print('xxx')
        
    def create_business(self, corporation, macro, macro_results, timeline, time_step_opt): 
        # add businesses
        name = 'Business_' + str(len(Business.instances))
        capacity  = 100
        
        estimate = Business('Estimate', 100, timeline, investments = macro_results['CAPEX'] * capacity, estimate = True)
        
        value = Valuation()
        DCF_EV = value.DCF_EV(estimate, macro, timeline, timeline, 0.1/12)
        PE_EV = value.PE_EV(estimate, macro, 10, timeline)
        IRR_yearly = value.IRR_business(estimate, macro, timeline, timeline, macro_results['CAPEX'] * capacity, 0.1/12, time_step_opt)
        del estimate        
       
        if corporation.human == True:
            print('Possible to invest in business named ' + name)
            print('Investment needed is :' + str(macro_results['CAPEX'] * capacity))
            print('EV of estimated business is :'+ str((DCF_EV+ PE_EV)/2))
            print('IRR is :' + str(IRR_yearly))
            answer = (input('Start business [y/n]: ').lower() == 'y')
        else:
            answer = False
        
        investments = 0
        if answer == True:
            print(corporation.name  + ' is now the owner of ' + name + '!')
            corporation.append_business(name, capacity, timeline, macro_results['CAPEX'] * capacity)
            investments = investments + macro_results['CAPEX'] * capacity
        return investments       


class RandomPopups(object):
    def __init__(self):
        #self.capacity_rand = random.random()
        self.capacity_rand = random.rand()
        # xxx

def test_functionA():

    timeline = datetime.datetime(2019,1,1)
    project1 = Business('Business_1', 100, timeline)
    project2 = Business('Business_2', 200, timeline)
    project3 = Business('Business_3', 200, timeline, fixed_assets = 209000)
    project4 = Business('Business_4', 200, timeline, fixed_assets = 209000)

    corp1 = Corporation('Double_Corp', timeline, [project1, project2],
        fixed_assets = 2000000, human = True)
    corp2 = Corporation('Double_Corp', timeline, [project3, project4], 
        fixed_assets = 2000000)
    
    print('xxx')

    macro = Macro(10)
    events = Events()
    sim = GlobalSim(timeline, macro, events, time_step_opt = 'month')
    
    sim.prog_time()
    sim.prog_time()
    sim.prog_time()
    sim.prog_time()

    print(project1.fin.bs.df)
    print(corp1.fin.bs.df)
    print(corp1.fin.pl.df)

def test_functionB():

    timeline = datetime.datetime(2019,1,1)
    #project = Business('Business', 100, timeline, fixed_assets = 209000)
    #project1 = Business('Business_1', 200, timeline)

    
    corp1 = Corporation('Double_Corp', timeline, [],
        fixed_assets = 2000000, human = True)
    print('xxx')

    macro = Macro(10)
    events = Events()
    sim = GlobalSim(timeline, macro, events, time_step_opt = 'month')
    
    sim.prog_time()
    sim.prog_time()
    sim.prog_time()
    sim.prog_time()
    
    for ii in Business.instances:
        print(ii.name)
        print(ii.fin.bs.df)
        print(ii.fin.pl.df)
        print(ii.fin.se.df)
        print(ii.fin.cf.df)
    
    print(corp1.fin.bs.df)
    print(corp1.fin.pl.df)
    print(corp1.fin.se.df)
    print(corp1.fin.cf.df)



if __name__ == '__main__':
    print('main')
    #test_functionA()
    test_functionB()
    
    

"""

DCF_EV = value.DCF_EV(project2, sim.timeline, sim.timeline, 0.1/12)
PE_EV = value.PE_EV(project2, 10, sim.timeline)
print(DCF_EV)
print(PE_EV)

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
      
"""