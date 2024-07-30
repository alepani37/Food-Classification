%ricerca configurazione iniziale migliore

%% Inizializzazione sistema
clear;
close all;
clc;
% DATASET
%dataset_dir='garbage_classification'; %dataset_folder_name
dataset_dir = 'prova_resized_2';

% FEATURES extraction methods
% 'sift' for sparse features detection (SIFT descriptors computed at  
% Harris-Laplace keypoints) or 'dsift' for dense features detection (SIFT
% descriptors computed at a grid of overlapped patches

%desc_name = 'sift';
%desc_name = 'dsift';
%desc_name = 'msdsift';
desc_name = 'color_sift';

% FLAGS
do_feat_extraction = 1;
do_split_sets = 1;
do_show_logs = 1;
do_show_internal_logs = 0; %per non vedere tutti i log strani che rallentano
do_form_codebook = 1;
do_feat_quantization = 1;

do_L2_NN_classification = 1;

do_visualize_feat = 1;
do_visualize_words = 1;
do_visualize_confmat = 1;
do_visualize_res = 1;
do_have_screen = 1; %~isempty(getenv('DISPLAY'));
do_chi2_NN_classification = 0;

do_svm_linar_classification = 1;
do_svm_llc_linar_classification = 0;
do_svm_precomp_linear_classification = 0;
do_svm_inter_classification = 0;
do_svm_chi2_classification = 0;

visualize_feat = 0;
visualize_words = 0;
visualize_confmat = 1;
visualize_res = 0;
%have_screen = ~isempty(getenv('DISPLAY'));
have_screen = 1;
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
num_train_img = 150; %numero per ogni classe
% number of images selected for test (e.g. 50 for Caltech-101)
num_test_img = 30;  %numero per ogni classe
% number of codewords (i.e. K for the k-means algorithm) %%CHANGE
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

if do_feat_extraction   
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


%% Load pre-computed SIFT features for test images (OBL)

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

if do_form_codebook
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


%% (OBL) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%   EXERCISE 1: K-means Descriptor quantization                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% K-means descriptor quantization means assignment of each feature
% descriptor with the identity of its nearest cluster mean, i.e.
% visual word. Your task is to quantize SIFT descriptors in all
% training and test images using the visual dictionary 'VC'
% constructed above.
%
% TODO:
% 1.1 compute Euclidean distances between VC and all descriptors
%     in each training and test image. Hint: to compute all-to-all
%     distance matrix for two sets of descriptors D1 & D2 use
%     dmat=eucliddist(D1,D2);
% 1.2 compute visual word ID for each feature by minimizing
%     the distance between feature SIFT descriptors and VC.
%     Hint: apply 'min' function to 'dmat' computed above along
%     the dimension (1 or 2) corresponding to VC, i.g.:
%     [mv,visword]=min(dmat,[],2); if you compute dmat as 
%     dmat=eucliddist(dscr(i).sift,VC);

if do_feat_quantization
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
disp("Sezione normalizzazione istogramma Eseguita")


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   End of EXERCISE 2                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% %%LLC Coding
if 1 %do_svm_llc_linar_classification
    parfor i=1:length(desc_train)
        %disp(desc_train(i).imgfname);
        desc_train(i).llc = max(LLC_coding_appr(VC,desc_train(i).sift)); %max-pooling
        desc_train(i).llc=desc_train(i).llc/norm(desc_train(i).llc); %L2 normalization
    end
    parfor i=1:length(desc_test) 
        %disp(desc_test(i).imgfname);
        desc_test(i).llc = max(LLC_coding_appr(VC,desc_test(i).sift));
        desc_test(i).llc=desc_test(i).llc/norm(desc_test(i).llc);
    end
end
%%%%end LLC coding
disp("Codifica LLC completata")

% Concatenate bof-histograms into training and test matrices 
bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);
if 1 %do_svm_llc_linar_classification
    llc_train = cat(1,desc_train.llc);
    llc_test = cat(1,desc_test.llc);
end


% Construct label Concatenate bof-histograms into training and test matrices 
labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);
disp("Construct label Concatenate bof-histograms into training and test matrices")

%% Sezione per la conversione in GPU array
bof_trainGPU = gpuArray(bof_train);
bof_testGPU = gpuArray(bof_test);
disp("Conversine in gpuArray completata")

%% SEZIONE DI CLASSIFICAZIONE

% NN classification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NN_L2
if 0 %do_L2_NN_classification
    % Compute L2 distance between BOFs of test and training images
    bof_l2dist=eucliddist(bof_test,bof_train);
    
    % Nearest neighbor classification (1-NN) using L2 distance
    [mv,mi] = min(bof_l2dist,[],2); %val, colonna nel train
    bof_l2lab = labels_train(mi);
    
    method_name='NN L2';

    %Calcolo metriche
    acc_NN_L2 = sum(bof_l2lab==labels_test)/length(labels_test);
    %fprintf('\n*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);
   
    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        bof_l2lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_NN_L2 = calculateMacroF1Score(ConfM, classes);

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    debug_inserisci_righe_al_csv("prima_analisi.csv", ...
        {method_name, nwords_codebook, acc_NN_L2, f1_NN_L2});
end
disp("Classificazione NN_L2 Eseguita");

%% NN_Chi2
if 0 %do_chi2_NN_classification
    % compute pair-wise CHI2
    bof_chi2dist = zeros(size(bof_test,1),size(bof_train,1));
    
    % bof_chi2dist = slmetric_pw(bof_train, bof_test, 'chisq');
    for i = 1:size(bof_test,1)
        for j = 1:size(bof_train,1)
            bof_chi2dist(i,j) = chi2(bof_test(i,:),bof_train(j,:)); 
        end
    end

    % Nearest neighbor classification (1-NN) using Chi2 distance
    [mv,mi] = min(bof_chi2dist,[],2);
    bof_chi2lab = labels_train(mi);

    method_name='NN Chi-2';

    %Calcolo metriche
    acc_NN_CHI2 =sum(bof_chi2lab==labels_test)/length(labels_test);
    %fprintf('*** %s ***\nAccuracy = %1.4f%% (classification)\n',method_name,acc*100);
    
    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        bof_chi2lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_NN_CHI2 = calculateMacroF1Score(ConfM, classes);

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    debug_inserisci_righe_al_csv("prima_analisi.csv", ...
        {method_name, nwords_codebook, acc_NN_CHI2, f1_NN_CHI2});
end
disp("Classificazione NN_CHI2 Eseguita");

%% LINEAR SVM
if 0 %do_svm_linar_classification
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


    % Calcolo metriche
    acc_SVM_LIN = calculateAccuracy(data,labels_test, ...
        svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);

    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_SVM_LIN = calculateMacroF1Score(ConfM, classes);

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    debug_inserisci_righe_al_csv("prima_analisi.csv", ...
        {method_name, nwords_codebook, acc_SVM_LIN, f1_SVM_LIN});
end
disp("Classificazione SVM_LIN Eseguita");


%% LLC LINEAR SVM
if 0 %do_svm_llc_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,llc_train,opt_string);
    end
    %select the best C among C_vals and test your model on the testing set.
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,llc_train,['-t 0 -c ' num2str(C_vals(ind))]);
    %disp('*** SVM - linear LLC max-pooling ***');
    svm_llc_lab=svmpredict(labels_test,llc_test,model);
    method_name= 'svm_lin_llc' %'llc+max-pooling';

    % Calcolo metriche
    %accuracy
    acc_SVM_LIN_LLC = calculateAccuracy(data,labels_test, ...
        svm_llc_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);

    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        svm_llc_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_SVM_LIN_LLC = calculateMacroF1Score(ConfM, classes);

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    debug_inserisci_righe_al_csv("prima_analisi.csv", ...
        {method_name, nwords_codebook, acc_SVM_LIN_LLC, f1_SVM_LIN_LLC});
end
disp("Classificazione SVM_LIN_LLC Eseguita");



%% try a non-linear svm with the histogram intersection kernel!

if 0 %do_svm_inter_classification
    Ktrain=zeros(size(bof_train,1),size(bof_train,1));
    for i=1:size(bof_train,1)
        for j=1:size(bof_train,1)
            hists = [bof_train(i,:);bof_train(j,:)];
            Ktrain(i,j)=sum(min(hists));
        end
    end

    Ktest=zeros(size(bof_test,1),size(bof_train,1));
    for i=1:size(bof_test,1)
        for j=1:size(bof_train,1)
            hists = [bof_test(i,:);bof_train(j,:)];
            Ktest(i,j)=sum(min(hists));
        end
    end

    % cross-validation
    C_vals=log2space(3,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... consider this if the kernel is computationally inefficient.
    %disp('*** SVM - intersection kernel ***');
    [precomp_ik_svm_lab,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);

    method_name='SVM_IK';
    
    % Calcolo metriche
    %accuracy
    acc_SVM_IK = calculateAccuracy(data,labels_test, ...
        precomp_ik_svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);

    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        precomp_ik_svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_SVM_IK = calculateMacroF1Score(ConfM, classes);

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    debug_inserisci_righe_al_csv("prima_analisi.csv", ...
        {method_name, nwords_codebook, acc_SVM_IK, f1_SVM_IK});
end
disp("Classificazione SVM_IK Eseguita");

%% SVM CHI2

if 0 %do_svm_chi2_classification    
    % compute kernel matrix
    Ktrain = kernel_expchi2(bof_train,bof_train);
    Ktest = kernel_expchi2(bof_test,bof_train);
    
    % cross-validation
    C_vals=log2space(2,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... 
    % consider this if the kernel is computationally inefficient.
    disp('*** SVM - Chi2 kernel ***');
    [precomp_chi2_svm_lab,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    
    method_name='SVM Chi2';

    % Calcolo metriche
    %accuracy
    acc_SVM_CHI2 = calculateAccuracy(data,labels_test, ...
        precomp_chi2_svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);

    % Compute classification accuracy and F1-Score
    ConfM = return_confusionmatrix(data,labels_test, ...
        precomp_chi2_svm_lab,classes, ...
        method_name,desc_test,...
        do_visualize_confmat & do_have_screen,... 
        do_visualize_res & do_have_screen);
    f1_SVM_CHI2 = calculateMacroF1Score(ConfM, classes);

    %Inserimento risultati nel csv di primaanalisi
    %Struttura:
    %classificatore_usato	n_codeword	acc	f1
    debug_inserisci_righe_al_csv("prima_analisi.csv", ...
        {method_name, nwords_codebook, acc_SVM_CHI2, f1_SVM_CHI2});

end
disp("Classificazione SVM_CHI2 Eseguita");