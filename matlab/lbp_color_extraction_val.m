function [trainLBP,testLBP, valLBP] = lbp_color_extraction_val(data,num_classes,num_istance_per_class_train,num_instance_per_class_test,num_instance_per_class_val, info)
    
    trainLBP_ = cell(num_classes,num_istance_per_class_train*3);
    testLBP_ = cell(num_classes,num_instance_per_class_test*3);
    valLBP_ = cell(num_classes,num_instance_per_class_val*3);
    for i = 1 : length(data)
        disp(length(data))
        %estrazione delle LBP per ogni immagine di train
        %fprintf("\nEstrazione elemento %d", num2str(i))
        i
        %estrazione immagini di train della i-esima classe
        img = data(i).files(data(i).train_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));

            R = im(:,:,1);
            G = im(:,:,2);
            B = im(:,:,3);

            % LBP extraction for each channel
            lbpR = extractLBPFeatures(R);
            lbpG = extractLBPFeatures(G);
            lbpB = extractLBPFeatures(B);

            fname =  fullfile(info.base, info.first,info.dsdir,data(i).classname,img(j));
            base_fname = regexprep(fname,['.' info.desc_name],'.jpg');

            trainLBP_{i, (j-1)*3 + 1}.filename = base_fname;
            trainLBP_{i, (j-1)*3 + 1}.hist = lbpR;

            trainLBP_{i, (j-1)*3 + 2}.filename = base_fname;
            trainLBP_{i, (j-1)*3 + 2}.hist = lbpG;

            trainLBP_{i, (j-1)*3 + 3}.filename = base_fname;
            trainLBP_{i, (j-1)*3 + 3}.hist = lbpB;
        end

        %estrazione immgini di test della i-esima classe
        img = data(i).files(data(i).test_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));

            % Split the image into color channels
            R = im(:,:,1);
            G = im(:,:,2);
            B = im(:,:,3);

            % LBP extraction for each channel
            lbpR = extractLBPFeatures(R);
            lbpG = extractLBPFeatures(G);
            lbpB = extractLBPFeatures(B);
            
            % Save the LBP histograms for each channel consecutively
            fname = fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j));
            base_fname = regexprep(fname, ['.' info.desc_name], '.jpg');

            testLBP_{i, (j-1)*3 + 1}.filename = base_fname;
            testLBP_{i, (j-1)*3 + 1}.hist = lbpR;

            testLBP_{i, (j-1)*3 + 2}.filename = base_fname;
            testLBP_{i, (j-1)*3 + 2}.hist = lbpG;

            testLBP_{i, (j-1)*3 + 3}.filename = base_fname;
            testLBP_{i, (j-1)*3 + 3}.hist = lbpB;
        end

        %estrazione immgini di validation della i-esima classe
        img = data(i).files(data(i).val_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));

            % Split the image into color channels
            R = im(:,:,1);
            G = im(:,:,2);
            B = im(:,:,3);

            % LBP extraction for each channel
            lbpR = extractLBPFeatures(R);
            lbpG = extractLBPFeatures(G);
            lbpB = extractLBPFeatures(B);
            
            % Save the LBP histograms for each channel consecutively
            fname = fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j));
            base_fname = regexprep(fname, ['.' info.desc_name], '.jpg');

            valLBP_{i, (j-1)*3 + 1}.filename = base_fname;
            valLBP_{i, (j-1)*3 + 1}.hist = lbpR;

            valLBP_{i, (j-1)*3 + 2}.filename = base_fname;
            valLBP_{i, (j-1)*3 + 2}.hist = lbpG;

            valLBP_{i, (j-1)*3 + 3}.filename = base_fname;
            valLBP_{i, (j-1)*3 + 3}.hist = lbpB;
        end
    end
    
    %cose x train
    num = 1;
    for i = 1 : size(trainLBP_,1)
        for j = 1 : size(trainLBP_,2)
            trainLBP(num).filename = trainLBP_{i,j}.filename;
            trainLBP(num).class = i;
            trainLBP(num).hist = trainLBP_{i,j}.hist;
            num = num + 1; 
        end
    end
    
    %cose x test
    num = 1;
    for i = 1 : size(testLBP_,1)
        for j = 1 : size(testLBP_,2)
            testLBP(num).filename = testLBP_{i,j}.filename;
            testLBP(num).class = i;
            testLBP(num).hist = testLBP_{i,j}.hist;
            num = num + 1; 
        end
    end

    %cose per validation
    num = 1;
    for i = 1 : size(valLBP_,1)
        for j = 1 : size(valLBP_,2)
            valLBP(num).filename = valLBP_{i,j}.filename;
            valLBP(num).class = i;
            valLBP(num).hist = valLBP_{i,j}.hist;
            num = num + 1; 
        end
    end
    %organizzazione decente delle LBP
    fprintf("Estrazione delle feature LBP completata correttamente\n");
end