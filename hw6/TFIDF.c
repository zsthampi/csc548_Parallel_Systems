// zsthampi -  Zubin S Thampi

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
#include <math.h>
#include "mpi.h"
#include <stddef.h>

#define MAX_WORDS_IN_CORPUS 32
#define MAX_FILEPATH_LENGTH 16
#define MAX_WORD_LENGTH 16
#define MAX_DOCUMENT_NAME_LENGTH 8
#define MAX_STRING_LENGTH 64

typedef char word_document_str[MAX_STRING_LENGTH];

typedef struct __attribute__((__packed__)) o {
	char word[32];
	char document[8];
	int wordCount;
	int docSize;
	int numDocs;
	int numDocsWithWord;
} obj;

typedef struct __attribute__((__packed__)) w {
	char word[32];
	int numDocsWithWord;
	int currDoc;
} u_w;

static int myCompare (const void * a, const void * b)
{
    return strcmp (a, b);
}

int main(int argc , char *argv[]){

	// MPI Init 
	MPI_Init(&argc, &argv);

	// Assign nproc and rank 
	int nproc, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int i,j;
	int numDocs, docSize, contains;
	char filename[MAX_FILEPATH_LENGTH], word[MAX_WORD_LENGTH], document[MAX_DOCUMENT_NAME_LENGTH];
	
	// Will hold all TFIDF objects for all documents
	obj TFIDF[MAX_WORDS_IN_CORPUS];
	int TF_idx = 0;
	
	// Will hold all unique words in the corpus and the number of documents with that word
	u_w unique_words[MAX_WORDS_IN_CORPUS];
	int uw_idx = 0;
	
	// Will hold the final strings that will be printed out
	word_document_str strings[MAX_WORDS_IN_CORPUS];

	// Create datatype for struct - obj 
	MPI_Datatype MPI_TFIDF;
	int block_lengths[6] = {32,8,1,1,1,1};
	MPI_Datatype types[6] = {MPI_CHAR,MPI_CHAR,MPI_INT,MPI_INT,MPI_INT,MPI_INT};
	MPI_Aint offsets[6];
	MPI_Aint base;

	MPI_Get_address(&TFIDF[0], &base);
	MPI_Get_address(&TFIDF[0].word, &offsets[0]);
	MPI_Get_address(&TFIDF[0].document, &offsets[1]);
	MPI_Get_address(&TFIDF[0].wordCount, &offsets[2]);
	MPI_Get_address(&TFIDF[0].docSize, &offsets[3]);
	MPI_Get_address(&TFIDF[0].numDocs, &offsets[4]);
	MPI_Get_address(&TFIDF[0].numDocsWithWord, &offsets[5]);
	offsets[0] -= base;
	offsets[1] -= base;
	offsets[2] -= base;
	offsets[3] -= base;
	offsets[4] -= base;
	offsets[5] -= base;

	MPI_Type_create_struct(6, block_lengths, offsets, types, &MPI_TFIDF);
	MPI_Type_commit(&MPI_TFIDF);

	// Create datatype for struct - u_w
	MPI_Datatype MPI_U_W;
	int block_lengths_u_w[3] = {32,1,1};
	MPI_Datatype types_u_w[3] = {MPI_CHAR,MPI_INT,MPI_INT};
	MPI_Aint offsets_u_w[3];
	MPI_Aint base_u_w;

	MPI_Get_address(&unique_words[0], &base_u_w);
	MPI_Get_address(&unique_words[0].word, &offsets_u_w[0]);
	MPI_Get_address(&unique_words[0].numDocsWithWord, &offsets_u_w[1]);
	MPI_Get_address(&unique_words[0].currDoc, &offsets_u_w[2]);
	offsets_u_w[0] -= base_u_w;
	offsets_u_w[1] -= base_u_w;
	offsets_u_w[2] -= base_u_w;

	MPI_Type_create_struct(3, block_lengths_u_w, offsets_u_w, types_u_w, &MPI_U_W);
	MPI_Type_commit(&MPI_U_W);
	
	if (rank == 0) {
		DIR* files;
		struct dirent* file;
	
		//Count numDocs
		if((files = opendir("input")) == NULL){
			printf("Directory failed to open\n");
			exit(1);
		}
		while((file = readdir(files))!= NULL){
			// On linux/Unix we don't want current and parent directories
			if(!strcmp(file->d_name, "."))	 continue;
			if(!strcmp(file->d_name, "..")) continue;
			numDocs++;
		}

		// Broadcast numDocs to all processes
		MPI_Bcast(&numDocs, 1, MPI_INT, 0, MPI_COMM_WORLD);
	} else {
		MPI_Bcast(&numDocs, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	if (rank == 0) {
		// Create a status object 
		MPI_Status status;
		int count;

		// MPI_Reduce(MPI_IN_PLACE, &TF_idx, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Gather(NULL, 0, MPI_TFIDF, &TFIDF, MAX_WORDS_IN_CORPUS, MPI_TFIDF, 0, MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);

		for (int proc = 1; proc < nproc; proc++) {
			MPI_Recv(&TFIDF[TF_idx], MAX_WORDS_IN_CORPUS, MPI_TFIDF, proc, proc, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_TFIDF, &count);
			TF_idx += count;
		}

		// Create a buffer for storing unique words from each rank 
		u_w unique_words_buffer[MAX_WORDS_IN_CORPUS];

		for (int proc = 1; proc < nproc; proc++) {
			// Receive the unique words from each rank 
			MPI_Recv(&unique_words_buffer, MAX_WORDS_IN_CORPUS, MPI_U_W, proc, proc, MPI_COMM_WORLD, &status);
			// Get the size of each 
			MPI_Get_count(&status, MPI_U_W, &count);

			// Iterate through each of them 
			for (i = 0; i < count; i++) {
				contains = 0;
				// If unique_words array at rank 0 already contains the word, just increment numDocsWithWord
				for(j = 0; j < uw_idx; j++) {
					if(!strcmp(unique_words[j].word, unique_words_buffer[i].word)){
						contains = 1;
						unique_words[j].numDocsWithWord += unique_words_buffer[i].numDocsWithWord;
						break;
					}
				}
				
				// If unique_words array does not contain it, make a new one with corresponding numDocsWithWord value
				if(!contains) {
					strcpy(unique_words[uw_idx].word, unique_words_buffer[i].word);
					unique_words[uw_idx].numDocsWithWord = unique_words_buffer[i].numDocsWithWord;
					uw_idx++;
				}
			}
		}

	} else {
		// Loop through each document and gather TFIDF variables for each word
		for(i=rank; i<=numDocs; i+=(nproc-1)){
			sprintf(document, "doc%d", i);
			sprintf(filename,"input/%s",document);
			FILE* fp = fopen(filename, "r");
			if(fp == NULL){
				printf("Error Opening File: %s\n", filename);
				exit(0);
			}
			
			// Get the document size
			docSize = 0;
			while((fscanf(fp,"%s",word))!= EOF)
				docSize++;
			
			// For each word in the document
			fseek(fp, 0, SEEK_SET);
			while((fscanf(fp,"%s",word))!= EOF){
				contains = 0;
				
				// If TFIDF array already contains the word@document, just increment wordCount and break
				for(j=0; j<TF_idx; j++) {
					if(!strcmp(TFIDF[j].word, word) && !strcmp(TFIDF[j].document, document)){
						contains = 1;
						TFIDF[j].wordCount++;
						break;
					}
				}
				
				//If TFIDF array does not contain it, make a new one with wordCount=1
				if(!contains) {
					strcpy(TFIDF[TF_idx].word, word);
					strcpy(TFIDF[TF_idx].document, document);
					TFIDF[TF_idx].wordCount = 1;
					TFIDF[TF_idx].docSize = docSize;
					TFIDF[TF_idx].numDocs = numDocs;
					TF_idx++;
				}
				
				contains = 0;
				// If unique_words array already contains the word, just increment numDocsWithWord
				for(j=0; j<uw_idx; j++) {
					if(!strcmp(unique_words[j].word, word)){
						contains = 1;
						if(unique_words[j].currDoc != i) {
							unique_words[j].numDocsWithWord++;
							unique_words[j].currDoc = i;
						}
						break;
					}
				}
				
				// If unique_words array does not contain it, make a new one with numDocsWithWord=1 
				if(!contains) {
					strcpy(unique_words[uw_idx].word, word);
					unique_words[uw_idx].numDocsWithWord = 1;
					unique_words[uw_idx].currDoc = i;
					uw_idx++;
				}
			}
			fclose(fp);
		}
		
		// MPI_Reduce(&TF_idx, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);
		// MPI_Gather(&TFIDF, TF_idx, MPI_TFIDF, NULL, 0, MPI_TFIDF, 0, MPI_COMM_WORLD);
		// MPI_Barrier(MPI_COMM_WORLD);

		// Send TFIDF values to rank 0
		MPI_Send(&TFIDF, TF_idx, MPI_TFIDF, 0, rank, MPI_COMM_WORLD);

		// Send the unique words array 
		MPI_Send(&unique_words, uw_idx, MPI_U_W, 0, rank, MPI_COMM_WORLD);
		
	}
	

	if (rank == 0) {
		// Print TF job similar to HW4/HW5 (For debugging purposes)
		printf("-------------TF Job-------------\n");
		for(j=0; j<TF_idx; j++)
			printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].wordCount, TFIDF[j].docSize);
			
		// Use unique_words array to populate TFIDF objects with: numDocsWithWord
		for(i=0; i<TF_idx; i++) {
			for(j=0; j<uw_idx; j++) {
				if(!strcmp(TFIDF[i].word, unique_words[j].word)) {
					TFIDF[i].numDocsWithWord = unique_words[j].numDocsWithWord;	
					break;
				}
			}
		}
		
		// Print IDF job similar to HW4/HW5 (For debugging purposes)
		printf("------------IDF Job-------------\n");
		for(j=0; j<TF_idx; j++)
			printf("%s@%s\t%d/%d\n", TFIDF[j].word, TFIDF[j].document, TFIDF[j].numDocs, TFIDF[j].numDocsWithWord);
			
		// Calculates TFIDF value and puts: "document@word\tTFIDF" into strings array
		for(j=0; j<TF_idx; j++) {
			double TF = 1.0 * TFIDF[j].wordCount / TFIDF[j].docSize;
			double IDF = log(1.0 * TFIDF[j].numDocs / TFIDF[j].numDocsWithWord);
			double TFIDF_value = TF * IDF;
			sprintf(strings[j], "%s@%s\t%.16f", TFIDF[j].document, TFIDF[j].word, TFIDF_value);
		}
		
		// Sort strings and print to file
		qsort(strings, TF_idx, sizeof(char)*MAX_STRING_LENGTH, myCompare);
		FILE* fp = fopen("output.txt", "w");
		if(fp == NULL){
			printf("Error Opening File: output.txt\n");
			exit(0);
		}
		for(i=0; i<TF_idx; i++)
			fprintf(fp, "%s\n", strings[i]);
		fclose(fp);
	}
	
	// MPI Finalize 
	MPI_Finalize();

	return 0;	
}
