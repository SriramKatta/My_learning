#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

struct Entry {
  int row;
  int col;
  double val;
  Entry(int r, int c, double v) : row(r), col(c), val(v) {}
};

std::vector<Entry> paralleldataload(std::ifstream &fin, int nnz, int rank,
                                    int size, int filesize, int data_offset) {
  int numlines = nnz / size + ((nnz % size) > rank ? 1 : 0);
  MPI_Offset chunksize = (filesize - data_offset) / size;
  std::string line;
  MPI_Offset currentoffset = data_offset + chunksize * rank;
  fin.seekg(currentoffset);

  MPI_Offset endoffset = currentoffset + chunksize;
  std::vector<Entry> localchunk;
  localchunk.reserve(numlines);

  MPI_Offset current_pos = fin.tellg(); // get current position
  if (current_pos > 0) {
    fin.seekg(current_pos - 1);
    char prev;
    fin.get(prev);
    if (prev != '\n') {
      std::getline(fin, line);
    }
  }

  while (fin.tellg() < endoffset) {
    std::getline(fin, line);
    if (line.empty()) {
      break;
    }
    int r, c;
    double v;
    if (sscanf(line.c_str(), "%d %d %lf", &r, &c, &v) == 3) {
      localchunk.emplace_back(r - 1, c - 1, v);
    }
  }
  return localchunk;
}

void printfilewithrank(std::vector<Entry> &vec, int rank) {
  std::string filename("rank");
  filename += '1' + rank;
  std::ofstream fout(filename);
  for (auto &[row, col, val] : vec) {
    fout << row << " " << col << " " << val << std::endl;
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 2) {
    if (rank == 0)
      std::cerr << "Usage: " << argv[0] << " matrix.mtx\n";
    MPI_Finalize();
    return 1;
  }

  std::string filename = argv[1];

  MPI_Offset data_offset = 0;
  MPI_Offset filesize = 0;
  size_t nrows = 0, ncols = 0, nnz = 0;
  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
  MPI_File_get_size(file, &filesize);
  MPI_File_close(&file);

  std::ifstream fin(filename);
  std::string line;

  data_offset = 0;
  while (std::getline(fin, line)) {
    data_offset += line.length() + 1; // +1 for newline character
    if (!line.empty() && line[0] != '%')
      break;
  }
  sscanf(line.c_str(), "%lu %lu %lu", &nrows, &ncols, &nnz);

  if (0 == rank) {
    printf("rows %ld, cols %ld, nnz : %ld\n", nrows, ncols, nnz);
  }

  auto start = MPI_Wtime();
  auto localchunk =
      paralleldataload(fin, nnz, rank, size, filesize, data_offset);
  auto end = MPI_Wtime();

  int localsize = localchunk.size();
  int globalsum = 0;

  MPI_Reduce(&localsize, &globalsum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (0 == rank) {
    printf("time taken : %lf\n", end - start);
    printf("nnz == globalsum %d\n", nnz == globalsum);
  }

  // printfilewithrank(localchunk, rank);

  MPI_Finalize();
  return 0;
}
