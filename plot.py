import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

class PlotData:
    def __init__(self, savedir):
        self.savedir = savedir
        self.blue = '#1F91BF'
        self.orange = '#FF6500'
        
        sns.set()
        sns.set_context("paper")
        sns.set(font_scale=1.3)
        sns.set_style("white")
        plt.rcParams.update({"text.usetex": True, "text.latex.preamble":[r"\usepackage{amsmath}",r"\usepackage{siunitx}",],})
        
    def plot_cdf(self, x, xname, plotname):
        q25 = np.quantile(x, 0.25)
        q50 = np.quantile(x, 0.5)
        q75 = np.quantile(x, 0.75)

        plt.figure()
        ax = sns.distplot(x, hist_kws={'cumulative': True}, kde_kws={'cumulative': True}, bins=np.arange(1,36))
        ax.axvline(q25, ymax = 0.25, linestyle=(0, (5, 3)))
        ax.axvline(q50, ymax = 0.5, linestyle=(0, (5, 2)))
        ax.axvline(q75, ymax = 0.75, linestyle=(0, (5, 1)))
        ax.annotate('25% ({:.0f})'.format(q25), xy=(q25, 0.25), xytext=(q25-6., 0.25))
        ax.annotate('50% ({:.0f})'.format(q50), xy=(q50, 0.50), xytext=(q50-6.35, 0.50))
        ax.annotate('75% ({:.0f})'.format(q75), xy=(q75, 0.75), xytext=(q75-6.6, 0.75))
        ax.set_xlabel(xname)
        ax.set_ylabel('Cumulative probability')
        ax.set_xlim((0, np.amax(x)))
        ax.set_ylim((0,1))
        plt.savefig(self.savedir+plotname+'_cdf.pdf', bbox_inches = "tight")
        plt.close()


class PlotResults:
    def __init__(self, savedir):
        self.savedir = savedir
        self.blue = '#1F91BF'
        self.orange = '#FF6500'
        self.color = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725) #seaborn blue
        self.gray = (0.5490196078431373, 0.5490196078431373, 0.5490196078431373) #seaborn gray
        self.ora = (0.8666666666666667, 0.5176470588235295, 0.3215686274509804) #seaborn orange
        
        sns.set()
        sns.set_context("paper")
        sns.set(font_scale=1.3)
        sns.set_style("white")
        plt.rcParams.update({"text.usetex": True, "text.latex.preamble":[r"\usepackage{amsmath}",r"\usepackage{siunitx}",],})

    
    def plot_training_metrics(self, model):
        # plot performance during training
        metrics = ['bce', 'roc-auc', 'pr-auc', 'precision', 'recall', 'f1']
        labels = ('train', 'val')
        
        x = model.history.epoch
        
        fig, axs = plt.subplots(int(len(metrics)/2)+1,2, sharex=True, figsize=(9,12))
        fig.subplots_adjust(wspace=.3)
        
        for i, m in enumerate(metrics):
            axs[i%3+1, int(i/3)].plot(x, model.metrics.metrics[m], label=labels[0], color=self.blue)
            axs[i%3+1, int(i/3)].plot(x, model.metrics.metrics['val_'+m], label=labels[1], color=self.orange)
            axs[i%3+1, int(i/3)].set_ylabel(m)
            axs[i%3+1, int(i/3)].set_xlim((0., np.amax(x)))
            axs[i%3+1, int(i/3)].set_xticks(x)
            
        #gs = axs[0,0].get_gridspec()
        axs[0,0].remove() ; axs[0,1].remove()
        ax = fig.add_subplot(int(len(metrics)/2)+1,1,1)
        ax.plot(x, model.history.history['loss'], label=labels[0], color=self.blue)
        ax.plot(x, model.history.history['val_loss'], label=labels[1], color=self.orange)
        ax.set_ylabel('Loss')
        ax.set_xlim((0., np.amax(x)))
        ax.set_xticks(x)

        ax.legend()
        axs[3,0].set_xlabel('epoch') ; axs[3,1].set_xlabel('epoch')
        plt.savefig(self.savedir+'learning_curve.pdf', bbox_inches = "tight")
        plt.close()
 
    def plot_training_metrics_comparison(self, results, names, plotname, a_base=0.1, a_delta=0.1):
        # plot performance during training
        metrics = ['bce', 'roc-auc', 'pr-auc', 'precision', 'recall', 'f1']
        labels = ('train', 'val')
        
        x = np.arange(1, results[0].shape[0]+1)
        
        fig, axs = plt.subplots(int(len(metrics)/2)+1,2, sharex=True, figsize=(9,12))
        fig.subplots_adjust(wspace=.3)
        for j, n in enumerate(names):
            for i, m in enumerate(metrics):
                axs[i%3+1, int(i/3)].plot(x, results[j][m], label=labels[0] + '- %s'%n, color=self.blue, alpha=a_delta*j+a_base)
                axs[i%3+1, int(i/3)].plot(x, results[j]['val_'+m], label=labels[1] + '- %s'%n, color=self.orange, alpha=a_delta*j+a_base)
                axs[i%3+1, int(i/3)].set_ylabel(m)
                axs[i%3+1, int(i/3)].set_xlim((0., np.amax(x)))
                axs[i%3+1, int(i/3)].set_xticks(x)
            
        #gs = axs[0,0].get_gridspec()
        axs[0,0].remove() ; axs[0,1].remove()
        ax = fig.add_subplot(int(len(metrics)/2)+1,1,1)
        for j, n in enumerate(names):
            ax.plot(x, results[j]['loss'], label=labels[0] + '- %s'%n, color=self.blue, alpha=a_delta*j+a_base)
            ax.plot(x, results[j]['val_loss'], label=labels[1] + '- %s'%n, color=self.orange, alpha=a_delta*j+a_base)
        ax.set_ylabel('Loss')
        ax.set_xlim((0., np.amax(x)))
        ax.set_xticks(x)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True, fontsize='xx-small')
        axs[3,0].set_xlabel('epoch') ; axs[3,1].set_xlabel('epoch')
        plt.savefig(self.savedir+plotname+'.pdf', bbox_inches = "tight")
        plt.close()
    
    def plot_training_metrics_comparison_batchsize(self, results, batchsizes):
        # plot performance during training
        metrics = ['bce', 'roc-auc', 'pr-auc', 'precision', 'recall', 'f1']
        labels = ('train', 'val')
        
        a_base = 0.1
        a_delta = 0.1
        
        x = np.arange(1, results[0].shape[0]+1)
        
        fig, axs = plt.subplots(int(len(metrics)/2)+1,2, sharex=True, figsize=(9,12))
        fig.subplots_adjust(wspace=.3)
        for j, b in enumerate(batchsizes):
            for i, m in enumerate(metrics):
                axs[i%3+1, int(i/3)].plot(x, results[j][m], label=labels[0] + ' b = %s'%b, color=self.blue, alpha=a_delta*j+a_base)
                axs[i%3+1, int(i/3)].plot(x, results[j]['val_'+m], label=labels[1] + ' b = %s'%b, color=self.orange, alpha=a_delta*j+a_base)
                axs[i%3+1, int(i/3)].set_ylabel(m)
                axs[i%3+1, int(i/3)].set_xlim((0., np.amax(x)))
                axs[i%3+1, int(i/3)].set_xticks(x)
            
        #gs = axs[0,0].get_gridspec()
        axs[0,0].remove() ; axs[0,1].remove()
        ax = fig.add_subplot(int(len(metrics)/2)+1,1,1)
        for j, b in enumerate(batchsizes):
            ax.plot(x, results[j]['loss'], label=labels[0] + ' b = %s'%b, color=self.blue, alpha=a_delta*j+a_base)
            ax.plot(x, results[j]['val_loss'], label=labels[1] + ' b = %s'%b, color=self.orange, alpha=a_delta*j+a_base)
        ax.set_ylabel('Loss')
        ax.set_xlim((0., np.amax(x)))
        ax.set_xticks(x)

        ax.legend()
        axs[3,0].set_xlabel('epoch') ; axs[3,1].set_xlabel('epoch')
        plt.savefig(self.savedir+'learning_curve_batchsizes.pdf', bbox_inches = "tight")
        plt.close()
    
    def plot_training_metrics_uncertainty(self, model):
        # plot performance during training
        metrics = ['bce', 'roc-auc', 'pr-auc', 'precision', 'recall', 'f1']
        labels = ('train', 'val')
        
        x = model.history.epoch
        
        fig, axs = plt.subplots(int(len(metrics)/2)+1,2, sharex=True, figsize=(9,12))
        fig.subplots_adjust(wspace=.3)
        
        for i, m in enumerate(metrics):
            axs[i%3+1, int(i/3)].plot(x, model.metrics.metrics_mean[m], label=labels[0], color=self.blue)
            axs[i%3+1, int(i/3)].fill_between(x, model.metrics.metrics_mean[m] - model.metrics.metrics_std[m], model.metrics.metrics_mean[m] + model.metrics.metrics_std[m], alpha=0.3, facecolor=self.blue)
            axs[i%3+1, int(i/3)].plot(x, model.metrics.metrics_mean['val_'+m], label=labels[1], color=self.orange)
            axs[i%3+1, int(i/3)].fill_between(x, model.metrics.metrics_mean['val_'+m] - model.metrics.metrics_std['val_'+m], model.metrics.metrics_mean['val_'+m] + model.metrics.metrics_std['val_'+m], alpha=0.3, facecolor=self.orange)
            axs[i%3+1, int(i/3)].set_ylabel(m)
            axs[i%3+1, int(i/3)].set_xlim((0., np.amax(x)))
            axs[i%3+1, int(i/3)].set_xticks(x)
            
        axs[0,0].remove() ; axs[0,1].remove()
        ax = fig.add_subplot(int(len(metrics)/2)+1,1,1)
        ax.plot(x, model.history.history['loss'], label=labels[0], color=self.blue)
        ax.plot(x, model.history.history['val_loss'], label=labels[1], color=self.orange)
        ax.set_ylabel('Loss')
        ax.set_xlim((0., np.amax(x)))
        ax.set_xticks(x)

        ax.legend()
        axs[3,0].set_xlabel('epoch') ; axs[3,1].set_xlabel('epoch')
        plt.savefig(self.savedir+'learning_curve.pdf', bbox_inches = "tight")
        plt.close()

    def plot_roc_curve(self, fpr, tpr, auc, cohort):
        plt.figure()
        plt.plot(fpr, tpr, label='AUC = %0.2f'%auc)
        plt.plot([0,1],[0,1], linestyle='--', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend()
        plt.savefig(self.savedir+'roc_curve-'+cohort+'.pdf', bbox_inches = "tight")
        plt.close()
        
    def plot_pr_curve(self, precision, recall, auc, cohort, balance):
        plt.figure()
        plt.plot(precision, recall, label='AUC = %0.2f'%auc)
        plt.plot([0,1],[balance, balance], linestyle='--', color='black')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve')
        plt.legend()
        plt.savefig(self.savedir+'pr_curve-'+cohort+'.pdf', bbox_inches = "tight")
        plt.close()
        
    def plot_roc_curves(self, fpr, tpr, auc, models):
        ls = (':','--','-.','-','-')
        plt.figure()
        plt.plot([0,1],[0,1], color=self.gray)
        for i, m in enumerate(models):
            c = self.color
            if i == 4:
                c = self.ora
            plt.plot(fpr[i], tpr[i], label= m + ' (AUC = %0.3f)'%auc[i], color=c, linestyle=ls[i])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic curve')
        plt.legend(fontsize=12)
        plt.savefig(self.savedir+'roc_curve-models.pdf', bbox_inches = "tight")
        plt.close()
        
    def plot_pr_curves(self, precision, recall, auc, balance, models):
        ls = (':','--','-.','-', '-')
        plt.figure()
        plt.plot([0,1],[balance, balance], color=self.gray)
        for i, m in enumerate(models):
            c = self.color
            if i == 4:
                c = self.ora
            plt.plot(precision[i], recall[i], label= m + ' (AUC = %0.3f)'%auc[i], color=c, linestyle=ls[i])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall curve')
        plt.legend(fontsize=12, loc='upper right')
        plt.savefig(self.savedir+'pr_curve-models.pdf', bbox_inches = "tight")
        plt.close()

    def plot_dist_performance(self, x, name, mu, beta, xlim=(-1,1)):
        plt.figure() 
        ax = sns.distplot(x) 
        ax.set_xlabel('Relative hospital performance $\{}_k$'.format(name)) 
        ax.set_ylabel('Hospitals')
        ax.set_xlim(xlim) 
        plt.legend(title=r"$\mu = {:.2f}$".format(mu)) 
        #plt.legend(title=r"$\mu = {:.2f}$ ; $\rho = {:.2f}$".format(mu, beta)) 
        plt.savefig(self.savedir+name+'s.pdf', bbox_inches = "tight")
        plt.close()

    def plot_dist_difference(self, x, A, B, name, A_name, B_name):
        min_plot = min(np.amin(x[A]), np.amin(x[B]))
        max_plot = max(np.amax(x[A]), np.amax(x[B]))
        
        corr = np.corrcoef(x[A], x[B])[0,1]

        plt.figure()
        sns.scatterplot(x[A], x[B], label=r"$\rho = {:.2f}$".format(corr))
        plt.plot(np.unique(x[A]), LinearRegression().fit(np.array(x[A]).reshape(-1,1), np.array(x[B]).reshape(-1,1)).predict(np.unique(x[A]).reshape(-1,1)), label='Linear trend', color=sns.color_palette()[1])
        plt.plot([min_plot, max_plot], [min_plot, max_plot], linestyle='dashed', color='gray', label='Identity')
        plt.xlabel('Relative hospital performance $\{}_k$ [{}]'.format(name, A_name))
        plt.ylabel('Relative hospital performance $\{}_k$ \n [{}]'.format(name, B_name))
        plt.legend()
        plt.savefig(self.savedir+name+'s_comparison.pdf', bbox_inches = "tight")
        plt.close()