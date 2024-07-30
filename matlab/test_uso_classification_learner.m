%Test per usare il classification learner
%Si crea il dataset con i metodi desiderati e poi lo si da in pasto al
%classification learner

%% Inizializzazione sistema
clear;
close all;
clc;

%SETTINGS

% DATASET
dataset_dir = 'prova_resized_bn_2';

% FEATURES extraction methods

desc_name = 'sift'; %sparse features detection (SIFT descriptors computed at Harris-Laplace keypoints)
%desc_name = 'dsift'; %SIFT % descriptors computed at a grid of overlapped patches
%desc_name = 'msdsift';
%desc_name = 'color_sift';

% FLAGS


% PATHS
basepath = '..';
wdir = pwd;
libsvmpath = [ wdir(1:end-6) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

% BOW PARAMETERS
max_km_iters = 1500; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

% number of images selected for training (e.g. 30 for Caltech-101)
num_train_img = 160; %numero per ogni classe
% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img = 40;  %numero per ogni classe
% number of codewords (i.e. K for the k-means algorithm)
nwords_codebook = 1000;
%NUmero massimo di immagini prendibili per ogni classe
num_max_img_per_classe = 200;

% image file extension
file_ext='jpg';


%% Create a new dataset split
file_split = 'split.mat';
if 1 %do_split_sets    
    data = create_dataset_split_structure_from_unbalanced_sets(...
        fullfile(basepath, 'img', dataset_dir), ... 
        num_train_img, ...
        num_test_img , ...
        file_ext, ...
        num_max_img_per_classe); %numero di immagini massimo da considerare per classe
    save(fullfile(basepath,'img',dataset_dir,file_split),'data');
else
    load(fullfile(basepath,'img',dataset_dir,file_split));
end
classes = {data.classname}; % create cell array of class name strings

disp("Immagini caricate correttamente")

%% Extract SIFT features fon training and test images

if 1   
    extract_sift_features(fullfile('..','img',dataset_dir),desc_name)    
end

disp("Estrazione delle feature SIFT completata correttamente")




%% Load pre-computed SIFT features for training images (OBL)

% The resulting structure array 'desc' will contain one
% entry per images with the following fields:
%  desc(i).r :    Nx1 array with y-coordinates for N SIFT features
%  desc(i).c :    Nx1 array with x-coordinates for N SIFT features
%  desc(i).rad :  Nx1 array with radius for N SIFT features
%  desc(i).sift : Nx128 array with N SIFT descriptors
%  desc(i).imgfname : file name of original image

%carica .sift di train
lasti=1;
for i = 1:length(data) %per ogni categoria trovata
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'train'); %ex. 0001.dsift
     for j = 1:length(images_descs) 
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,images_descs{j});
        if 0 %do_show_logs
            fprintf('Loading %s \n',fname);
        end
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_train(lasti)=tmp.desc;
        desc_train(lasti).sift = single(desc_train(lasti).sift);
        lasti=lasti+1;
    end;
end;
disp("Caricamento completato")

%carica .sift di test
lasti=1;
for i = 1:length(data)
     images_descs = get_descriptors_files(data,i,file_ext,desc_name,'test');
     for j = 1:length(images_descs) 
        fname = fullfile(basepath,'img',dataset_dir,data(i).classname,images_descs{j});
        fprintf('Loading %s \n',fname);
        tmp = load(fname,'-mat');
        tmp.desc.class=i;
        tmp.desc.imgfname=regexprep(fname,['.' desc_name],'.jpg');
        desc_test(lasti)=tmp.desc;
        desc_test(lasti).sift = single(desc_test(lasti).sift);
        lasti=lasti+1;
    end;
end;


%% Build visual vocabulary using k-means (OBL) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1
    fprintf('\nBuild visual vocabulary:\n');

    % concatenate all descriptors from all images into a n x d matrix 
    DESC = [];
    labels_train = cat(1,desc_train.class);
    for i=1:length(data) %per ogni categoria delle immagini da classificare
        desc_class = desc_train(labels_train==i);
        randimages = randperm(num_train_img);
        randimages = randimages(1:5); %perché prende solo le prime 5?
        DESC = vertcat(DESC,desc_class(randimages).sift);
        %a caso vengono prese le sift di 
    end

    % sample random M (e.g. M=20,000) descriptors from all training descriptors
    r = randperm(size(DESC,1));
    r = r(1:min(length(r),nfeat_codebook));

    DESC = DESC(r,:);

    % run k-means
    K = nwords_codebook; % size of visual vocabulary
    fprintf('running k-means clustering of %d points into %d clusters...\n',...
        size(DESC,1),K)
    % input matrix needs to be transposed as the k-means function expects 
    % one point per column rather than per row

    % form options structure for clustering
    cluster_options.maxiters = max_km_iters;
    cluster_options.verbose  = 1;

    [VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
    VC = VC';%transpose for compatibility with following functions
    %clear DESC;
end

%% Quantizzazione descrittori con k-emans

if 1
    fprintf('\nFeature quantization (hard-assignment)...\n');
    %quantizzazione per le immagini di train
    for i=1:length(desc_train)  
      sift = desc_train(i).sift(:,:);
      dmat = eucliddist(sift,VC); %distanza euclidea in 128-dimensioni
      %dmat mi dice la distanza del descrittore del keypoint da ogni
      %keyword.
      [quantdist,visword] = min(dmat,[],2); %prendo keyword + vicina
      % save feature labels
      desc_train(i).visword = visword;
      desc_train(i).quantdist = quantdist;
    end

    for i=1:length(desc_test)    
      sift = desc_test(i).sift(:,:); 
      dmat = eucliddist(sift,VC);
      [quantdist,visword] = min(dmat,[],2);
      % save feature labels
      desc_test(i).visword = visword;
      desc_test(i).quantdist = quantdist;
    end
end

%% Represent each image by the normalized histogram of visual (OBL)
% word labels of its features. Compute word histogram H over 
% the whole image, normalize histograms w.r.t. L1-norm.
%
% TODO:
% 2.1 for each training and test image compute H. Hint: use
%     Matlab function 'histc' to compute histograms.


N = size(VC,1); % number of visual words

parfor i=1:length(desc_train) 
    visword = desc_train(i).visword; 
    %visword = visualword associata ad ogni keypoint dell'img

    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)
    if norm_bof_hist
        H = H/sum(H);
    end
  
    % save histograms
    desc_train(i).bof=H(:)';
end

parfor i=1:length(desc_test) 
    visword = desc_test(i).visword;
    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)
    if norm_bof_hist
        H = H/sum(H);
    end
  
    % save histograms
    desc_test(i).bof=H(:)';
end
disp("Sezione normalizzazione istogramma BoF")



% Concatenate bof-histograms into training and test matrices 
bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);


% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);



disp("Pronto per la fase d classificazione");

%Test di classificazion con SVM

%% LINEAR SVM
if 1 %do_svm_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,bof_train,opt_string);
    end
    %select the best C among C_vals and test your model on the testing set.
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,bof_train,['-t 0 -c ' num2str(C_vals(ind))]);
    disp('*** SVM - linear ***');
    svm_lab=svmpredict(labels_test,bof_test,model);
    
    method_name='SVM linear';
    
    do_visualize_confmat = 1; 
    do_have_screen = 1;
    do_visualize_res = 1;

    
    % Calcolo metriche
    acc_SVM_LIN = calculateAccuracy(data,labels_test, ...
        svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen)

    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_SVM_LIN = calculateMacroF1Score(ConfM, classes)

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    
end
disp("Classificazione SVM_LIN Eseguita");

%% CLassification Learner
%Adesso si può andare sull'app -> Tab in alt -> Apps -> Classification
%Learner

%Parametri (su Learn)
%Data set Variable: bof_train
%   Use columsn as variables
%Response:
%   From workspace -> labels_train

%Parametri (Test)
%Test daata -> From workspace ->  bof_test



%Il dataset di training nè bof_train e quello di test è bof_test