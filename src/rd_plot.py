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
    legenda["gain"] = {}


    legenda["gain"]["colore"] = [palette[5],'-']
    legenda["gain"]["legends"] = "gain [1]"
    legenda["gain"]["symbols"] = ["*"]*3
    legenda["gain"]["markersize"] = [5]*3



    legenda["base"]["colore"] = [palette[0],'-']
    legenda["base"]["legends"] = "reference [4]"
    legenda["base"]["symbols"] = ["o"]*3
    legenda["base"]["markersize"] = [5]*3

    legenda["our"]["colore"] = [palette[3],'-']
    legenda["our"]["legends"] = "proposed"

    legenda["our"]["symbols"] = ["*"]*len(psnr_res["our"])
    legenda["our"]["markersize"] = [5]*len(psnr_res["our"])

    
    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    list_names = list(bpp_res.keys()) #[base our]



    minimo_bpp, minimo_psnr = 10000,1000
    massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 
        print("type_name: ",type_name)

        bpp =bpp_res[type_name][::-1]
        psnr = psnr_res[type_name][::-1]
        colore = legenda[type_name]["colore"][0]
        leg = legenda[type_name]["legends"]


        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=7)
        #for x, y, marker, markersize_t in zip(bpp, psnr, symbols, markersize):
        if type_name == "our":

            #plt.plot(bpp, psnr, marker="o", markersize=7, color =  colore)
            plt.plot(bpp[-1], psnr[-1], marker="*", markersize=10, color =  colore)
            plt.plot(bpp[:-1], psnr[:-1], marker="o", markersize=7, color =  colore)
        elif type_name == "base":
            plt.plot(bpp, psnr, marker="*", markersize=10, color =  colore)
        else:
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

    minimo = min(0.1, int(minimo_bpp*10))*10
    bpp_tick =   [round(x)/10 for x in range(1, int(massimo_bpp*10 + 1))]
    plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='lower right', fontsize = 25)



    plt.grid(True)
    plt.savefig("rebuttal2.png")
    #wandb.log({"Compression":epoch,
    #          "Compression/rate distorsion trade-off": wandb.Image(plt)})
    #plt.close()  




def main():

    psnr_res = {}
    bpp_res = {}



    psnr_res["our"] =   [30.937, 30.342, 29.79, 29.374, 28.74, 27.23, 26.21]
    bpp_res["our"] =  [0.325, 0.274, 0.236, 0.21, 0.179, 0.14, 0.0952] 


    bpp_res["gain"] = [0.342, 0.2024, 0.09]
    psnr_res["gain"] = [29.256,27.582, 25.9183]  


    psnr_res["base"] =  [
      27.581536752297392,
      29.196703405493214,
      30.972162072759534,

    ]

    bpp_res["base"] = [0.13129340277777776,
      0.20889282226562503,
      0.3198581271701389]
    plot_rate_distorsion(bpp_res, psnr_res,0)


if __name__ == "__main__":  
    main()