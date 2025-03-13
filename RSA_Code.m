
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   RSA code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subject_ids={'Sbj01' ,'Sbj02','Sbj03'};

nsubjects=numel(subject_ids);

for subject_num=1:nsubjects
    subject_id=subject_ids{subject_num};

    ds=cosmo_fmri_dataset(data_fn,'mask',mask_fn,...
                                'targets',[1 2 3 4 5 6],'chunks',1); 

    labels = {'Self_HonestT';'Self_MoralT';'Self_OtherT'; 'Other_HonestT'; 'Other_MoralT';'Other_OtherT';};
    ds.sa.labels = labels;

    cosmo_check_dataset(ds);
    
    fprintf('Dataset input:\n');
    cosmo_disp(ds);

    nsamples=size(ds.samples,1);
    model=zeros(nsamples);
    
    % Prepare the theoretical RDMs
    SelfHonestT_rdm = [0	1	1	1	1	1
                       1	0	0	0	0	0
                       1	0	0	0	0	0
                       1	0	0	0	0	0
                       1	0	0	0	0	0
                       1	0	0	0	0	0];

    SelfMoralT_rdm = [0	 1	0	0	0	0
                      1	 0	1	1	1	1
                      0	 1	0	0	0	0
                      0	 1	0	0	0	0
                      0	 1	0	0	0	0
                      0	 1	0	0	0	0];

    SelfOtherT_rdm = [0  0	1	0	0	0
                      0	 0	1	0	0	0
                      1	 1	0	1	1	1
                      0	 0	1	0	0	0
                      0	 0	1	0	0	0
                      0	 0	1	0	0	0];

    OtherHonestT_rdm = [0	0	0	1	0	0
                        0	0	0	1	0	0
                        0	0	0	1	0	0
                        1	1	1	0	1	1
                        0	0	0	1	0	0
                        0	0	0	1	0	0];

    OtherMoralT_rdm = [0	0	0	0	1	0
                        0	0	0	0	1	0
                        0	0	0	0	1	0
                        0	0	0	0	1	0
                        1	1	1	1	0	1
                        0	0	0	0	1	0];

    OtherOtherT_rdm = [0	0	0	0	0	1
                        0	0	0	0	0	1
                        0	0	0	0	0	1
                        0	0	0	0	0	1
                        0	0	0	0	0	1
                        1	1	1	1	1	0];

    % Compute neural RDMs
    % Univariate RDM 
    num_conditions=length(labels);
    options.metric = 'euclidean'; 
    univariate_rdm = cosmo_dissimilarity_matrix_measure(ds, options);
    idx = 1;  univariate_rdm_matrix = zeros(6,6);  
    for kk=1:length(univariate_rdm.samples)
        univariate_rdm_matrix(univariate_rdm.sa.targets1(kk),univariate_rdm.sa.targets2(kk))=univariate_rdm.samples(kk);
        univariate_rdm_matrix(univariate_rdm.sa.targets2(kk),univariate_rdm.sa.targets1(kk))=univariate_rdm.samples(kk);
    end

    Mean_6cons=mean(ds.samples,2);
 
    % Multivariate RDM 
    multivariate_rdm = zeros(num_conditions, num_conditions);
    for i = 1:num_conditions
        for j = 1:num_conditions
            if i ~= j
                corr_value = corr(ds.samples(i, :)', ds.samples(j, :)');
                multivariate_rdm(i, j) = 1 - corr_value; 
            end
        end
    end


    % Correlate neural RDMs with theoretical RDMs
    rdm_size = size(SelfHonestT_rdm, 1); 

    SelfHonestT_vec = SelfHonestT_rdm(triu(true(rdm_size), 1));
    SelfMoralT_vec  = SelfMoralT_rdm(triu(true(rdm_size), 1));
    SelfOtherT_vec  = SelfOtherT_rdm(triu(true(rdm_size), 1));
    OtherHonestT_vec = OtherHonestT_rdm(triu(true(rdm_size), 1));
    OtherMoralT_vec  = OtherMoralT_rdm(triu(true(rdm_size), 1));
    OtherOtherT_vec  = OtherOtherT_rdm(triu(true(rdm_size), 1));
    Identity_vec    = Identity_rdm(triu(true(rdm_size), 1));

    univariate_vec = univariate_rdm_matrix(triu(true(rdm_size), 1));
    multivariate_vec = multivariate_rdm(triu(true(rdm_size), 1));
    
    SelfHonestT_Univariate_corr(subject_num) = corr(univariate_vec, SelfHonestT_vec, 'Type', 'Spearman');
    SelfMoralT_Univariate_corr(subject_num)  = corr(univariate_vec, SelfMoralT_vec, 'Type', 'Spearman');
    SelfOtherT_Univariate_corr(subject_num)  = corr(univariate_vec, SelfOtherT_vec, 'Type', 'Spearman');
    OtherHonestT_Univariate_corr(subject_num) = corr(univariate_vec, OtherHonestT_vec, 'Type', 'Spearman');
    OtherMoralT_Univariate_corr(subject_num)  = corr(univariate_vec, OtherMoralT_vec, 'Type', 'Spearman');
    OtherOtherT_Univariate_corr(subject_num)  = corr(univariate_vec, OtherOtherT_vec, 'Type', 'Spearman');
    
    SelfHonestT_Multivariate_corr(subject_num) = corr(multivariate_vec, SelfHonestT_vec, 'Type', 'Spearman');
    SelfMoralT_Multivariate_corr(subject_num)  = corr(multivariate_vec, SelfMoralT_vec, 'Type', 'Spearman');
    SelfOtherT_Multivariate_corr(subject_num)  = corr(multivariate_vec, SelfOtherT_vec, 'Type', 'Spearman');
    OtherHonestT_Multivariate_corr(subject_num) = corr(multivariate_vec, OtherHonestT_vec, 'Type', 'Spearman');
    OtherMoralT_Multivariate_corr(subject_num)  = corr(multivariate_vec, OtherMoralT_vec, 'Type', 'Spearman');
    OtherOtherT_Multivariate_corr(subject_num)  = corr(multivariate_vec, OtherOtherT_vec, 'Type', 'Spearman');
    

    fisher_z = @(r) 0.5 * log((1 + r) / (1 - r));
    SelfHonestT_Univariate_z(subject_num) = fisher_z(SelfHonestT_Univariate_corr(subject_num));
    SelfMoralT_Univariate_z(subject_num)  = fisher_z(SelfMoralT_Univariate_corr(subject_num));
    SelfOtherT_Univariate_z(subject_num)  = fisher_z(SelfOtherT_Univariate_corr(subject_num));
    
    SelfHonestT_Multivariate_z(subject_num) = fisher_z(SelfHonestT_Multivariate_corr(subject_num));
    SelfMoralT_Multivariate_z(subject_num)  = fisher_z(SelfMoralT_Multivariate_corr(subject_num));
    SelfOtherT_Multivariate_z(subject_num)  = fisher_z(SelfOtherT_Multivariate_corr(subject_num));

%%

    univariate_threshold = 0.15; 
    multivariate_threshold = 0.15;
    
    Univariate_corr=[SelfHonestT_Univariate_corr(subject_num),...
                    SelfMoralT_Univariate_corr(subject_num),...
                    SelfOtherT_Univariate_corr(subject_num),...
                    OtherHonestT_Univariate_corr(subject_num),...
                    OtherMoralT_Univariate_corr(subject_num),...
                    OtherOtherT_Univariate_corr(subject_num)];
    
   Multivariate_corr=[SelfHonestT_Multivariate_corr(subject_num),...
                     SelfMoralT_Multivariate_corr(subject_num),...
                     SelfOtherT_Multivariate_corr(subject_num),...
                     OtherHonestT_Multivariate_corr(subject_num),...
                     OtherMoralT_Multivariate_corr(subject_num),...
                     OtherOtherT_Multivariate_corr(subject_num)];

    rdm={SelfHonestT_rdm,SelfMoralT_rdm,SelfOtherT_rdm,...
         OtherHonestT_rdm,OtherMoralT_rdm,OtherOtherT_rdm};
    
%%
    single_max_corr=0;
    subplot(4, 6, 9);
    univariate_union = zeros(num_conditions,num_conditions); 
    for i = 1:num_conditions
        if Univariate_corr(i) >= univariate_threshold
            univariate_union = univariate_union|rdm{i}; 
        end
        if Univariate_corr(i) == max(Univariate_corr)
            univariate_SingleMaxMdoel = rdm{i}; 
        end
    end
    univariate_union_vec = univariate_union(triu(true(rdm_size), 1));
    if ~all(univariate_union(:) == 0)
        univariate_corr_result = corr(univariate_union_vec, univariate_vec, 'Type', 'Spearman');
        single_max_corr = max(Univariate_corr); 
    end

    if single_max_corr >= univariate_corr_result
        univariate_model_maxCorr{subject_num} = univariate_SingleMaxMdoel.*single_max_corr;
    elseif single_max_corr < univariate_corr_result
        univariate_model_maxCorr{subject_num} = univariate_union.*univariate_corr_result;
    end

%%
    single_max_corr=0;
    subplot(4, 6, 11);
    multivariate_union = zeros(num_conditions,num_conditions); 
    for i = 1:num_conditions
        if Multivariate_corr(i) >= multivariate_threshold
            multivariate_union = multivariate_union|rdm{i};
        end
        if Multivariate_corr(i) == max(Multivariate_corr)
            multivariate_SingleMaxMdoel = rdm{i}; 
        end
    end
    multivariate_union_vec = multivariate_union(triu(true(rdm_size), 1));
    if ~all(multivariate_union(:) == 0)
        multivariate_corr_result = corr(multivariate_union_vec, multivariate_vec, 'Type', 'Spearman');
        single_max_corr = max(Multivariate_corr); 
    end

    if single_max_corr >= multivariate_corr_result
        multivariate_model_maxCorr{subject_num} = multivariate_SingleMaxMdoel.*single_max_corr;
    elseif single_max_corr < multivariate_corr_result
        multivariate_model_maxCorr{subject_num} = multivariate_union.*multivariate_corr_result;
    end


end

result = zeros(6, 6);
for i = 1:numel(univariate_model_maxCorr)
    result = result + univariate_model_maxCorr{i};
end
Final_mean_univariate_model_maxCorr = result / numel(univariate_model_maxCorr);

result = zeros(6, 6);
for i = 1:numel(multivariate_model_maxCorr)
    result = result + multivariate_model_maxCorr{i};
end
Final_mean_multivariate_model_maxCorr = result / numel(multivariate_model_maxCorr);
