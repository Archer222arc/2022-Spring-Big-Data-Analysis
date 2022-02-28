function maketable(data,label,opt)
% this function prints a table in latex
if ~isfield(label,"tltle2");    col_label = 1;  else;   col_label=0;end
if ~isfield(opt,"filename");    filename = "table"; end
if ~isfield(opt,"H");   opt.H = "H";    end
[m,n] = size(data);
H = opt.H;
filename = opt.filename;

title1 = label.title1;
length1 = label.length1;
title_c = label.col;

data = num2cell(data);
title1 = num2cell(title1);
length1 = num2cell(length1);
title_c = num2cell(title_c);

global table
table = fopen(filename+".tex","w");

fprintf(table,"\\begin{table}");
if H ~= "None"
    fprintf(H);
end

if isfield(opt,"label")
    fprintf(table,"\\label{"+string(opt.label)+"}\n");
end
fprintf(table,"\\centering\n");
fprintf(table,"\\begin{tabular}{");
for i = 1 : n+1
    fprintf(table,"|c");
end
fprintf(table,"|}\n");

% fprintf(table,"\\begin{tabular}");
if col_label == 1
    title2 = label.title2;
    title2 = num2cell(title2);
    hline;
    fprintf(table,"\\multirow{%d}{*}{ "+string(title1{1})+"}",col_label+1);
    for i = 2:length(title1)
        fprintf(table," &\\multicolumn{%d}{c|}{"+string(title1{i})+"}",length1{i});
    end
    fprintf(table,"\\\\\\cline{2-"+string(n+1)+"}\n");
    for i = 1:length(title2)
        fprintf(table," &"+string(title2{i}));
    end
    wrap;
    for i = 1:m
        fprintf(table,title_c{i});    
        for j = 1:n
            fprintf(table," & $%3.2e$",data{i,j});
        end
        wrap;
    end
else
    hline;
    for i = 1:n
        fprintf(table,string(title1{i}));
        if i < n
            fprintf(table," &");
        end
    end
    wrap;
    for i = 1:m
        fprintf(table,title_c{i});    
        for j = 1:n
            fprintf(table," & $%3.2e$",data{i,j});
        end
        wrap;
    end
end
fprintf(table,"\\end{tabular}\n");
if isfield(opt,"caption")
    fprintf(table,"\\caption{"+string(opt.caption)+"}\n");
end
fprintf(table,"\\end{table}\n");
fclose(table);



function wrap
    fprintf(table,"\\\\\\hline\n");
end
function hline
    fprintf(table,"\n\\hline\n");
end
   
end
