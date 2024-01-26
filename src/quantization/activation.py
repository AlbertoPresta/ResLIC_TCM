
import torch
import torch.nn as nn



class NonSymStanH(nn.Module):
    def __init__(self, beta,  num_sigmoids,  extrema = 5, trainable =True, ):
        super(NonSymStanH, self).__init__()
        #print("non-linear-sum")
        self.num_sigmoids = int(num_sigmoids)
        self.beta = beta
        self.extrema = extrema     
        self.minimo = - extrema 
        self.massimo = extrema
            
        
        self.range_num = torch.arange(self.minimo  + 0.5 ,self.massimo ).type(torch.FloatTensor)
        if self.num_sigmoids > 0:
            self.jump = len(self.range_num)/self.num_sigmoids
            self.levels = num_sigmoids + 1
        
        else:
            self.levels = extrema*2 + 1 
        # bias 
            
        
        if self.num_sigmoids == 0:
            self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor), requires_grad= trainable) # quantizzazione allenabile (ha senso)?
        else:
                #self.b = torch.nn.Parameter(torch.FloatTensor(num_sigmoids).normal_().sort()[0]) # punti a caso
            c = len(self.range_num)/self.num_sigmoids
            self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))


        if self.num_sigmoids == 0:
            self.w = torch.nn.Parameter(torch.ones(len(self.range_num)), requires_grad= trainable )
        else:
            self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  )      


    
        self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        #print("trainable parameters for the quantizer: ",self.tr_parameters)


        self.length = len(self.range_num) if self.num_sigmoids ==0 else self.num_sigmoids 
        #print("lunghezza---> ",self.length)

        self.map_sos_cdf = {}
        self.map_cdf_sos = {}




        n = (torch.sum(self.w)/2).item()
        self.cum_w = torch.zeros(self.length + 1)
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
        self.cum_w = torch.sub(self.cum_w,n)
        #print("CUMULATIVE WEIGHTS ARE: ",self.cum_w)


        self.calculate_average_points()
        self.calculate_distance_points()

        self.update_state()

        


    def update_state(self, device = torch.device("cuda")):
        self.update_cumulative_weights(device = device )
        self.calculate_average_points( ) #self.average_points
        self.average_points = self.average_points.to(device)
        self.calculate_distance_points() #self.distance_points
        self.distance_points = self.distance_points.to(device)
        self.define_channels_map()



    def calculate_average_points(self):
        self.average_points = torch.add(self.cum_w[1:], self.cum_w[:-1])/2
        

    def calculate_distance_points(self):
        self.distance_points = torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
        


    def update_cumulative_weights(self,device = torch.device("cuda")):
        #if self.num_sigmoids == 0:
        n = (torch.sum(self.w)/2).item()
        self.cum_w = torch.zeros(self.length + 1).to(self.w.device)
        self.cum_w[0] = 0.0
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)
        self.cum_w = torch.sub(self.cum_w,n) # -  self.extrema 
        self.cum_w = self.cum_w.to(device)


    def reinitialize_weights_and_bias(self):
        if self.num_sigmoids == 0:
            self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )
            self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))
        else:
            self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 
            c = len(self.range_num)/self.num_sigmoids
            self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))
        






    def define_channels_map(self):
        mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
        map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
        map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
        self.map_sos_cdf = map_float_to_int
        self.map_cdf_sos = map_int_to_float

        #print("maps: ",self.map_sos_cdf)

        #self.mapping_decoding = pd.DataFrame(list(self.map_cdf_sos.items()), columns=['key', 'value'])
        #self.mapping_decoding.set_index('key', inplace=True)
        #print("***************************** mapping fatto")
    
    


    def f(self,x):
        return 2*torch.sigmoid(2*x) - 1

    def forward(self, x, beta=None):
        #if self.trainable_bias:
        b = torch.sort(self.b)[0] # non serve ?
        #else:
        #    b = self.b   
         
        if beta is not None:
            if beta == -1:
                return torch.sum(self.w[:,None]*torch.relu(torch.sign(x - b[:,None])) - self.w[:,None]/2,dim = 1).unsqueeze(1)             
                #return torch.stack([self.w[i]*(torch.relu(torch.sign(x-b[i]))) - self.w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
            else:
                return torch.sum((self.w[:,None]/2)*self.f(beta*(x - b[:,None])),dim = 1).unsqueeze(1)
                #return torch.stack([(self.w[i]/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
        else:
            return torch.sum( (self.w[:,None]/2)* self.f(self.beta*(x - b[:,None]))  ,dim = 1).unsqueeze(1)
            #return torch.stack([(self.w[i]/2)*self.f(self.beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0)






class SymStanH(nn.Module):
    def __init__(self, beta,  num_sigmoids, extrema = 5, trainable = True):
        super(SymStanH, self).__init__()


        self.num_sigmoids = int(num_sigmoids)
        self.beta = beta


   
        self.minimo = - extrema 
        self.massimo = extrema
            
        
        self.range_num = torch.arange(0.5 ,self.massimo ).type(torch.FloatTensor)
        if self.num_sigmoids > 0:
            self.jump = len(self.range_num)/self.num_sigmoids
            self.levels = num_sigmoids + 1
        
        else:
            self.levels = extrema*2 + 1 





        
        if self.num_sigmoids == 0:

            self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor),requires_grad= trainable)

            self.w = torch.nn.Parameter(torch.ones(len(self.range_num)),requires_grad= trainable )  # + torch.relu( torch.randn(len(self.range_num)))  #torch.relu( torch.randn(len(self.range_num))) +  torch.relu( torch.randn(len(self.range_num))) +   torch.relu( torch.randn(len(self.range_num)))  +  torch.relu( torch.randn(len(self.range_num))) + torch.relu( torch.randn(len(self.range_num)))
        else:
            c = len(self.range_num)/self.num_sigmoids
            self.b = torch.nn.Parameter(torch.arange( self.jump/2   ,self.massimo + self.jump/2 , c),requires_grad= trainable)

            
            self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump,requires_grad= trainable  ) 

    

    
        self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("count number of parameters for the quantizer: ",self.tr_parameters )


        self.length = len(self.range_num) if self.num_sigmoids ==0 else self.num_sigmoids 


        self.map_sos_cdf = {}
        self.map_cdf_sos = {}

        self.update_state()


    def update_weights(self):
        self.sym_w =  torch.cat((torch.flip(self.w,[0]),self.w),0)
        self.sym_b = torch.cat((torch.flip(-self.b,[0]),self.b),0) 

    def update_state(self, device = torch.device("cuda")):
        self.update_weights()
        self.update_cumulative_weights( )
        self.cum_w = self.cum_w.to(device)
        self.calculate_average_points( ) #self.average_points
        self.average_points = self.average_points.to(device)
        self.calculate_distance_points() #self.distance_points
        self.distance_points = self.distance_points.to(device)

        



    def update_cumulative_weights(self):
        self.cum_w = torch.zeros(self.length + 1)
        self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
        self.cum_w = torch.cat((-torch.flip(self.cum_w[1:], dims = [0]),self.cum_w),dim = 0)






    def reinitialize_weights_and_bias(self):
        if self.num_sigmoids == 0:
            self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )
            self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))
        else:
            self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 
            c = len(self.range_num)/self.num_sigmoids
            self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))
        


    def define_channels_map_prova(self ):

        minimum = -int(self.cum_w.shape[0])//2
        maximum = -minimum
        mapping = torch.arange(minimum, maximum, 1).numpy()
        map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
        map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
        self.map_sos_cdf = map_float_to_int
        self.map_cdf_sos = map_int_to_float
        print("----------------------------------------------------------------------------------------------------")
        print("sos cdf: ", self.map_sos_cdf)
        print("----------------------------------------------------------------------------------------------------")      
        print("cdf sos: ", self.map_cdf_sos) 





    def define_channels_map(self ):


        mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
        map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
        map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
        self.map_sos_cdf = map_float_to_int
        self.map_cdf_sos = map_int_to_float
        #print("----------------------------------------------------------------------------------------------------")
        #print("sos cdf: ", self.map_sos_cdf)
        #print("----------------------------------------------------------------------------------------------------")      
        #print("cdf sos: ", self.map_cdf_sos)       
    
    def calculate_average_points(self):
        self.average_points =  torch.add(self.cum_w[1:], self.cum_w[:-1])/2


    def calculate_distance_points(self):
        self.distance_points =   torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
       

    def f(self,x):
        return 2*torch.sigmoid(2*x) - 1

    def forward(self, x, beta=None):
        b = torch.sort(self.sym_b)[0]
        if beta is not None:
            if beta == -1:
                return torch.sum((self.sym_w[:,None].to(x.device)/2)*(torch.sign(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)               
                #return torch.stack([w[i]*(torch.relu(torch.sign(x-b[i]))) - w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
            else:
                return torch.sum((self.sym_w[:,None].to(x.device)/2)*self.f(beta*(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)
                #return torch.stack([(self.w[i]/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
        else:
            return torch.sum((self.sym_w[:,None]/2)* self.f(self.beta*(x - b[:,None]))  ,dim = 1).unsqueeze(1).to(x.device)
        
