filename = "./dataset/gisette_scale";
file = fopen(filename);
l = fgetl(file);
A_train = zeros(6000,5000);
y_train = zeros(6000,1);
x = 0;
while l ~= -1
    x = x+1;
    str = strsplit(l,' ');
    for i = 2:length(str)
        sstr = split(str{i},':');
        if length(sstr) < 2
            continue
        end
        A_train(x,str2double(sstr{1})) = str2double(sstr{2});
    end
    y_train(x) = str2double(str{1});
    l = fgetl(file);
end

filename = filename+".t";
file = fopen(filename);
l = fgetl(file);
A_test = zeros(1000,5000);
y_test = zeros(1000,1);
x = 0;
while l ~= -1
    x = x+1;
    str = strsplit(l,' ');
    for i = 2:length(str)
        sstr = split(str{i},':');
        if length(sstr) < 2
            continue
        end
        A_test(x,str2double(sstr{1})) = str2double(sstr{2});
    end
    y_test(x) = str2double(str{1});
    l = fgetl(file);
end

save("./dataset/gisette_train.mat","A_train");
save("./dataset/gisette_train_y.mat","y_train");
save("./dataset/gisette_test.mat","A_test");
save("./dataset/gisette_test_y.mat","y_test");

