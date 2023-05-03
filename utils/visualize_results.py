import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from utils.metrics import metrics, MMD
import contextily as ctx

label_names = ['barley', 'wheat', 'rapeseed', 'corn', 'sunflower', 'orchards',
       'nuts', 'permanent_meadows', 'temporary_meadows']

def plot_confusion_matrix(cm, classes, num_classes=9, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams.update({'font.size': 14})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize='large')
    plt.xlabel('Predicted label', fontsize='large')

    
def extract_features(model, dataloader, device):
    model.eval()
    
    test_predictions = []
    test_targets = []
    test_embeddings = torch.zeros((0, 64), dtype=torch.float32)
    
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true, field_id = batch

                embeddings, logits = model(x.to(device))
                preds = torch.argmax(logits, dim=1)
                test_predictions.extend(preds.detach().cpu().tolist())
                test_targets.extend(y_true.detach().cpu().tolist())
                test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)

            test_embeddings = np.array(test_embeddings)
            test_targets = np.array(test_targets)
            test_predictions = np.array(test_predictions)
            
            print('Extracted embedded features shape', test_embeddings.shape)
        return test_embeddings, test_targets, test_predictions


def plot2Dpca(zone, source_zone, target_zone, test_embeddings, test_targets, test_predictions, save_plot = False):
    pca = PCA(n_components=2)
    pca.fit(test_embeddings)
    pca_proj = pca.transform(test_embeddings)
    pca.explained_variance_ratio_

    # 2D Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 9
    label_names = ['barley', 'wheat', 'rapeseed', 'corn', 'sunflower', 'orchards',
       'nuts', 'permanent_meadows', 'temporary_meadows']
    
    for lab in range(num_categories):
        indices = test_targets ==lab
        ax.scatter(pca_proj[indices,0],
                   pca_proj[indices,1], 
                   s = 12,
                   c=np.array(cmap(lab)).reshape(1,4), 
                   label = label_names[lab],
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    title = str('2D Features '+zone+' - Model: source '+source_zone+' - target '+target_zone)
    plt.title(title)
    plt.xlabel("First Component")
    plt.ylabel("Second Component")
    plt.grid()
    plt.show()
    if save_plot:
        fig.savefig('results/2Dfeature_pca_model'+source_zone+target_zone+'test'+zone+'.png')
    return pca_proj

def plot3Dpca(zone, source_zone, target_zone, test_embeddings, test_targets, test_predictions, save_plot = False):
    # compute projections on 3D space with PCA
    pca = PCA(n_components=3)
    pca.fit(test_embeddings)
    pca_proj = pca.transform(test_embeddings)
    pca.explained_variance_ratio_
    
    # 3D Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    num_categories = 9
    label_names = ['barley', 'wheat', 'rapeseed', 'corn', 'sunflower', 'orchards',
       'nuts', 'permanent_meadows', 'temporary_meadows']
    
    for lab in range(num_categories):
        indices = test_targets == lab
        ax.scatter(pca_proj[indices,0],
                   pca_proj[indices,1],
                   pca_proj[indices,2],
                   s = 10,
                   c=np.array(cmap(lab)).reshape(1,4), 
                   label = label_names[lab],
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    ax.set_xlabel('First Component', fontsize=10)
    ax.set_ylabel('Second Component', fontsize=10)
    ax.set_zlabel('Third Component', fontsize=10)
    title = str('3D Features '+zone+' - Model: source '+source_zone+' - target '+target_zone)
    plt.title(title)
    plt.show()
    if save_plot:
        fig.savefig('results/3Dfeature_pca_model'+source_zone+target_zone+'test'+zone+'.png')
    return pca_proj


def plot3D_source_target(source_zone, target_zone, 
                     source_embeddings, source_targets, source_predictions,
                     target_embeddings, target_targets, target_predictions,
                     image_name = None,
                     save_plot = True):
    # compute projections on 3D space with PCA
    pca_source = PCA(n_components=3)
    pca_source.fit(source_embeddings)
    pca_source_proj = pca_source.transform(source_embeddings)
    pca_source.explained_variance_ratio_
    
    pca_target = PCA(n_components=3)
    pca_target.fit(target_embeddings)
    pca_target_proj = pca_target.transform(target_embeddings)
    pca_target.explained_variance_ratio_
    
    # 3D Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    num_categories = 9
    label_names = ['barley', 'wheat', 'rapeseed', 'corn', 'sunflower', 'orchards',
       'nuts', 'permanent_meadows', 'temporary_meadows']
    
    for lab in range(num_categories):
        indices = source_targets == lab
        scatter1 = ax.scatter(pca_source_proj[indices,0],
                       pca_source_proj[indices,1],
                       pca_source_proj[indices,2],
                       s = 5,
                       #c=np.array(cmap(lab)).reshape(1,4), 
                       c='green',
                       #label = label_names[lab],
                       label = source_zone,
                       alpha=0.4)
        
    for lab in range(num_categories):
        indices = target_targets == lab
        scatter2 = ax.scatter(pca_target_proj[indices,0],
                       pca_target_proj[indices,1],
                       pca_target_proj[indices,2],
                       s = 5,
                       #c=np.array(cmap(lab)).reshape(1,4), 
                       c='blue',
                       #label = label_names[lab],
                       label = target_zone,
                       alpha=0.4)   
    
    ax.legend((scatter1, scatter2),('source','target'),title="Domains", fontsize='large', markerscale=2)
    
    ax.set_xlabel('First Component', fontsize=10)
    ax.set_ylabel('Second Component', fontsize=10)
    ax.set_zlabel('Third Component', fontsize=10)
    title = str('3D Features Comparison - Model: source '+source_zone+' - target '+target_zone)
    plt.title(title, fontsize = 15)
    plt.show()
    
    if save_plot:
        fig.savefig('results/'+image_name+'.png')
        
def plot_on_map():
    fig,axs = plt.subplots(1,2, figsize=(24,12))

    ax = axs[0]
    france_xlim = (-777823.199830,  1027313.660153)
    france_ylim = (5043620.874369, 6613943.183460)
    ax.set_xlim(*france_xlim)
    ax.set_ylim(*france_ylim)

    ctx.add_basemap(ax)

    ymin, xmin, ymax, xmax = field_parcels_geodataframe.to_crs(epsg=3857).total_bounds
    #ax.plot([xmin,xmin,xmax,xmax, xmin],[ymin,ymax,ymax,ymin, ymin])
    ax.plot([ymin],[xmin],"ro", markersize=20)
    ax.set_title("Ille-et-Vilaine within France")

    ax = axs[1]
    ax = field_parcels_geodataframe_dann.to_crs(epsg=3857).plot(column="classname", ax=ax, legend=False)
    ax.set_title("Ille-et-Vilaine")
    ax.set_xlim(*(-175000,  -155000))
    ax.set_ylim(*(6160000, 6180000))
    ctx.add_basemap(ax)