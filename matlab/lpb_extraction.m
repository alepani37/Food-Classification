function [trainLBP,testLBP] = lpb_extraction(data,num_classes,num_istance_per_class_train,num_instance_per_class_test,info)
    
    trainLBP_ = cell(num_classes,num_istance_per_class_train);
    testLBP_ = cell(num_classes,num_instance_per_class_test);
    %trainLBP = zeros(num_classes*num_istance_per_class_train,1);
    %testLBP = zeros(num_classes * num_instance_per_class_test,1);

    for i = 1 : length(data)
        disp(length(data))
        %estrazione delle LBP per ogni immagine di train
        %fprintf("\nEstrazione elemento %d", num2str(i))
        i
        %estrazione immagini di train della i-esima classe
        img = data(i).files(data(i).train_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));
            %conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            %lbp extraction
            lbp_img = extractLBPFeatures(im);
            fname =  fullfile(info.base, info.first,info.dsdir,data(i).classname,img(j));
            trainLBP_{i,j}.filename = regexprep(fname,['.' info.desc_name],'.jpg');
            trainLBP_{i,j}.hist = lbp_img;
        end

        %estrazione immgini di test della i-esima classe
        img = data(i).files(data(i).test_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));
            %conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            %lbp extraction
            lbp_img = extractLBPFeatures(im);
            fname =  fullfile(info.base, info.first,info.dsdir,data(i).classname,img(j));
            testLBP_{i,j}.filename = regexprep(fname,['.' info.desc_name],'.jpg');
            testLBP_{i,j}.hist = lbp_img;
        end
    end
    
    num = 1;
    for i = 1 : size(trainLBP_,1)
        for j = 1 : size(trainLBP_,2)
            trainLBP(num).filename = trainLBP_{i,j}.filename;
            trainLBP(num).class = i;
            trainLBP(num).hist = trainLBP_{i,j}.hist;
            num = num + 1; 
        end
    end

    num = 1;
    for i = 1 : size(testLBP_,1)
        for j = 1 : size(testLBP_,2)
            testLBP(num).filename = testLBP_{i,j}.filename;
            testLBP(num).class = i;
            testLBP(num).hist = testLBP_{i,j}.hist;
            num = num + 1; 
        end
    end
    %organizzazione decente delle LBP
    fprintf("Estrazione delle feature LBP completata correttamente\n");
end