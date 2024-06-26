clc;
clear;

% Leggi il file CSV
filename = 'Results__2024_06_02_11_39_47.csv';
opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
data = readtable(filename, opts);

% Estrai le colonne
sources = data.Sources;
keywords = data.Keywords;
total_weights = data.('Total Weights'); % Utilizza il nome originale della colonna
ind_weight0 = find(total_weights == 0); % Rimuove le righe dove Total Weights è uguale a 0.
sources(ind_weight0) = [];
keywords(ind_weight0) = [];
total_weights(ind_weight0) = [];

% Trova gli elementi unici per sources e keywords
unique_sources = unique(sources);
unique_keywords = unique(keywords);

% Realizzazione grafo bipartito
G = graph(sources, keywords, total_weights);
pesi = G.Edges.Weight;
h = plot(G, 'LineWidth', (0.2*pesi));%, 'EdgeLabel', pesi
h.NodeLabel = {};

% Imposta coordinate X fisse e coordinate Y equidistanti
coordinateX = [];
coordinateY = [];
nodes = G.Nodes.Name;
n_s = length(unique_sources);
n_k = length(unique_keywords);

% Calcolo delle coordinate Y equidistanti
source_step = 1 / (n_s + 1);
keyword_step = 1 / (n_k + 1);

source_y_values = source_step:source_step:(n_s * source_step);
keyword_y_values = keyword_step:keyword_step:(n_k * keyword_step);

source_idx = 1;
keyword_idx = 1;

for i = 1:numnodes(G)
    if ismember(nodes{i}, unique_sources)
        coordinateX = [coordinateX 1];
        coordinateY = [coordinateY source_y_values(source_idx)];
        source_idx = source_idx + 1;
    else
        coordinateX = [coordinateX 2];
        coordinateY = [coordinateY keyword_y_values(keyword_idx)];
        keyword_idx = keyword_idx + 1;
    end
end

h.XData = coordinateX;
h.YData = coordinateY;

% Imposta le etichette dei nodi
labelnode(h, 1:numnodes(G), ''); % Rimuove le etichette da tutti i nodi
for i = 1:numnodes(G)
    if ~ismember(nodes{i}, unique_sources) % Se il nodo è una keyword
        labelnode(h, i, nodes{i}); % Etichetta con il nome della keyword
    end
end

saveas(h, 'bipartite_graph.png');

% Inizializza la matrice di affiliazione
aff_matrix = zeros(length(unique_sources), length(unique_keywords));

% Popola la matrice di affiliazione con i total weights
for i = 1:height(sources)
    % Trova gli indici per la source e la keyword correnti
    source_idx = find(strcmp(unique_sources, sources{i}));
    keyword_idx = find(strcmp(unique_keywords, keywords{i}));
    
    % Assegna il valore del total weight alla matrice di affiliazione
    aff_matrix(source_idx, keyword_idx) = total_weights(i);
end

% Visualizza la matrice di affiliazione
disp('Affiliation matrix:');
disp(aff_matrix);

% Visualizza la matrice di affiliazione con i nomi di sorgenti e keywords
% Visualizza la matrice di affiliazione
disp('        Keywords       ');
disp(unique_keywords');
disp('   -----------------------------------');
for i = 1:length(unique_sources)
    fprintf('%-20s | ', unique_sources{i});
    fprintf('%8.2f ', aff_matrix(i, :));
    fprintf('\n');
end

% Opzionale: Salva la matrice di affiliazione in un file
%writematrix(aff_matrix, 'adjacency_matrix.csv');

% Calcolo della matrice di omofilia (proiezione nello spazio delle sources)
omo_matrix = aff_matrix * aff_matrix';

% Visualizzazione del risultato
disp('Matrice di omofilia:');
disp(omo_matrix);

% Realizzazione grafo relativo alla proieazione sulle sources del grafo
% bipartito
G1 = graph(omo_matrix, 'omitselfloop');

% Visualizza il grafo
figure;
h1 = plot(G1, 'Layout', 'force', 'NodeColor', 'k', 'NodeLabel', {});
%higlight(h1, 'NoeColor', 'r');
saveas(h1, 'sources_projection.png');


% Calcolo della matrice binaria a partire da quella di affiliazione 
%bin_aff_matrix = aff_matrix > 0;

% Calcolo della matrice relativa alla proiezione sulle keywords
key_matrix = aff_matrix' * aff_matrix;

% Visualizzazione del risultato
disp('Matrice proiezione sulle keywords:');
disp(key_matrix);

% Realizzazione grafo relativo alla proieazione sulle keywords del grafo
% bipartito
G2 = graph(key_matrix, 'omitselfloop');

% Visualizza il grafo
figure;
h2 = plot(G2, 'Layout', 'force', 'EdgeLabel', round(G2.Edges.Weight, 2));

% Modifica le etichette dei nodi del grafo G2
labelnode(h2, 1:numnodes(G2), unique_keywords);

saveas(h2, 'keywords_projection.png');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calcolo della matrice dicotomica a partire da quella di affiliazione 
bin_aff_matrix = aff_matrix > 0;

% Calcolo della matrice relativa alla proiezione sulle keywords con matrice
% dicotomica
key_matrix2 = bin_aff_matrix' * bin_aff_matrix;

% Visualizzazione del risultato
disp('Matrice proiezione sulle keywords (SECONDA VERSIONE):');
disp(key_matrix2);

% Realizzazione grafo relativo alla proieazione sulle keywords con matrice
% dicotomica
G3 = graph(key_matrix2, 'omitselfloop');

% Visualizza il grafo
figure;
h3 = plot(G3, 'Layout', 'force', 'EdgeLabel', round(G3.Edges.Weight, 2));

% Modifica le etichette dei nodi del grafo G3
labelnode(h3, 1:numnodes(G3), unique_keywords);


% Definizione delle matrici di adiacenza e grafi per il calcolo delle
% distanze
A = omo_matrix;
B = key_matrix;
C = key_matrix2;
A_dist = 1./A;
A_dist(A_dist==Inf) = 0;
B_dist = 1./B;
B_dist(B_dist==Inf) = 0;
C_dist = 1./C;
C_dist(C_dist==Inf) = 0;
G1_dist = graph(A_dist,'omitselfloops');
G2_dist = graph(B_dist,'omitselfloops');
G3_dist = graph(C_dist,'omitselfloops');

%Calcolo della distanza: se G è un grafo pesato il comando distances tiene 
% conto dei pesi degli archi
distanzaA = distances(G1_dist);
distanzaB = distances(G2_dist);
distanzaC = distances(G3_dist);

% Misure di nodo
% Grado
%Grado dei nodi del grafo relativo alla proiezione sulle sorgenti
disp("Matrice A:");
disp(A);
gradoA = centrality(G1,'degree','Importance', G1.Edges.Weight);

%Grado dei nodi del grafo relativo alla proiezione sulle keywords
disp("Matrice B:");
disp(B);
gradoB = centrality(G2,'degree','Importance', G2.Edges.Weight);

%Grado dei nodi del grafo relativo alla proiezione sulle keywords CON
%MATRICE DICOTOMICA
disp("Matrice C:");
disp(C);
gradoC = centrality(G3,'degree','Importance', G3.Edges.Weight);


% Betweenness non normalizzata
betaA = centrality(G1_dist,'betweenness','Cost',G1_dist.Edges.Weight);
betaB = centrality(G2_dist,'betweenness','Cost',G2_dist.Edges.Weight);
betaC = centrality(G3_dist,'betweenness','Cost',G3_dist.Edges.Weight);

% Misure di rete
%DENSITA'
N1 = numnodes(G1_dist);
M1 = numedges(G1_dist);
delta1 = 2*M1/((N1*N1)-N1);

N2 = numnodes(G2_dist);
M2 = numedges(G2_dist);
delta2 = 2*M2/((N2*N2)-N2);

N3 = numnodes(G3_dist);
M3 = numedges(G3_dist);
delta3 = 2*M3/((N3*N3)-N3);

%DIAMETRO
diametro1 = max(max(distanzaA));
diametro2 = max(max(distanzaB));
diametro3 = max(max(distanzaC));

%media delle misure di nodo
meanGradoA = mean(gradoA);
meanGradoB = mean(gradoB);
meanGradoC = mean(gradoC);

meanBetaA = mean(betaA);
meanBetaB = mean(betaB);
meanBetaC = mean(betaC);

meanGammaA = mean(gammaA);
meanGammaB = mean(gammaB);
meanGammaC = mean(gammaC);

%Barplot dei gradi
figure;
%bar(unique_sources, gradoA)
[sortedDegreeA, idxs] = sort(gradoA, 'Descend');
bar(unique_sources(idxs(1:20)), sortedDegreeA (1:20));
ylabel('Degree');
xtickangle(90);

figure;
[sortedDegreeB, idxs] = sort(gradoB, 'Descend');
bar(unique_keywords(idxs), sortedDegreeB);
ylabel('Degree');

figure;
[sortedDegreeC, idxs] = sort(gradoC, 'Descend');
bar(unique_keywords(idxs), sortedDegreeC);
ylabel('Degree');

%Barplot dei valori di betweenness
figure;
[sortedBetaA, idxs] = sort(betaA, 'Descend');
bar(unique_sources(idxs(1:20)), sortedBetaA (1:20));
ylabel('Betweenness');
xtickangle(90);

figure;
[sortedBetaB, idxs] = sort(betaB, 'Descend');
bar(unique_keywords(idxs), sortedBetaB);
ylabel('Betweenness');

figure;
[sortedBetaC, idxs] = sort(betaC, 'Descend');
bar(unique_keywords(idxs), sortedBetaC);
ylabel('Betweenness');