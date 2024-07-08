
import torch 
import wandb


def plot_sos(model, device,epoch,symmetry = False,lv = 0,n = 1000):


    if symmetry is False:
        x_min = float((min(model.gaussian_conditional[lv].stanh.b) + min(model.gaussian_conditional[lv].stanh.b)*0.5).detach().cpu().numpy())
        x_max = float((max(model.gaussian_conditional[lv].stanh.b)+ max(model.gaussian_conditional[lv].stanh.b)*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min - 30, x_max -30, step)
        x_values = x_values.repeat(model.gaussian_conditional[lv].M,1,1)
                
        y_values=model.gaussian_conditional[lv].stanh(x_values.to(device))[0,0,:]
        data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])

        log_dict = { "StanH":epoch,
                    "StanH/quantizer_soft_"+str(lv) : wandb.plot.line(table, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(model.gaussian_conditional[lv].stanh.beta))
                    
        }

        wandb.log(log_dict)

        
        y_values= model.gaussian_conditional[lv].stanh(x_values.to(device), -1)[0,0,:]
        data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        log_dict = { "StanH":epoch,
                    "StanH/actual_quantizer_" +str(lv): wandb.plot.line(table_inf, "x", "sos")
                    
        }
        wandb.log(log_dict)
    else:

        minimo = min(model.gaussian_conditional[lv].stanh.sym_b)
        massimo = max(model.gaussian_conditional[lv].stanh.sym_b)
        x_min = float((minimo + minimo*0.5).detach().cpu().numpy())
        x_max = float((massimo+ massimo*0.5).detach().cpu().numpy())
        step = (x_max-x_min)/n
        x_values = torch.arange(x_min - 30, x_max -30, step)
        x_values = x_values.repeat(model.gaussian_conditional[lv].M,1,1)
                
        y_values=model.gaussian_conditional[lv].stanh(x_values.to(device))[0,0,:]
        data = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table = wandb.Table(data=data, columns = ["x", "sos"])

        log_dict = { "StanH":epoch,
                    "StanH/quantizer_soft_": wandb.plot.line(table, "x", "sos", title='GaussianSoS/Gaussian SoS  with beta = {}'.format(model.gaussian_conditional[lv].stanh.beta))
                    
        }

        wandb.log(log_dict)

        
        y_values= model.gaussian_conditional.stanh(x_values.to(device), -1)[0,0,:]
        data_inf = [[x, y] for (x, y) in zip(x_values[0,0,:],y_values)]
        table_inf = wandb.Table(data=data_inf, columns = ["x", "sos"])
        log_dict = { "StanH":epoch,
                    "StanH/actual_quantizer": wandb.plot.line(table_inf, "x", "sos")
                    
        }
        wandb.log(log_dict)





import seaborn as sns
# Imposta la palette "tab10" di Seaborn
palette = sns.color_palette("tab10")
#rc('text', usetex=True)
#rc('font', family='Times New Roman')
import matplotlib.pyplot as plt
import wandb

def plot_rate_distorsion(bpp_res, psnr_res,epoch):


    legenda = {}
    legenda["base"] = {}
    legenda["our"] = {}



    legenda["base"]["colore"] = [palette[0],'-']
    legenda["base"]["legends"] = "reference"
    legenda["base"]["symbols"] = ["*"]*3
    legenda["base"]["markersize"] = [5]*3

    legenda["our"]["colore"] = [palette[3],'-']
    legenda["our"]["legends"] = "proposed"
    print("la lunghezza è questa----> ",len(psnr_res["our"]))
    legenda["our"]["symbols"] = ["*"]*len(psnr_res["our"])
    legenda["our"]["markersize"] = [5]*len(psnr_res["our"])

    
    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(bpp_res.keys()) #[base our]



    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp =bpp_res[type_name][::-1]
        psnr = psnr_res[type_name][::-1]
        colore = legenda[type_name]["colore"][0]
        leg = legenda[type_name]["legends"]

        bpp = torch.tensor(bpp).cpu().numpy()
        psnr = torch.tensor(psnr).cpu().numpy()

        
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=7)
        plt.plot(bpp, psnr, marker="o", markersize=7, color =  colore)
                



        for j in range(len(bpp)):
            if bpp[j] < minimo_bpp:
                minimo_bpp = bpp[j]
            if bpp[j] > massimo_bpp:
                massimo_bpp = bpp[j]
            
            if psnr[j] < minimo_psnr:
                minimo_psnr = psnr[j]
            if psnr[j] > massimo_psnr:
                massimo_psnr = psnr[j]

    minimo_psnr = int(minimo_psnr)
    massimo_psnr = int(massimo_psnr)
    psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
    plt.ylabel('PSNR', fontsize = 30)
    plt.yticks(psnr_tick)

    #print(minimo_bpp,"  ",massimo_bpp)

    bpp_tick =   [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 1))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    wandb.log({"Compression":epoch,
              "Compression/rate distorsion trade-off": wandb.Image(plt)})
    plt.close()  
