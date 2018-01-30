#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))
#define myabs(x,y)  (((x) > (y))? ((x)-(y)) : ((y)-(x)))

#if !defined(point) 
#define point 9
#endif

#if point == 5
#define  kernel(A)  A[(t+1)%2][y][x] = 0.125 * (A[t%2][y+1][x] - 2.0 * A[t%2][y][x] + A[t%2][y-1][x]) + \
									   0.125 * (A[t%2][y][x+1] - 2.0 * A[t%2][y][x] + A[t%2][y][x-1]) + \
									   A[t%2][y][x];
#define XSLOPE 1
#define YSLOPE 1
#define DATA_TYPE double
#elif point == 9
#define  kernel(A) A[(t+1)%2][y][x] =  0.96 * A[t%2][y][x] + \
									   0.0051 * (A[t%2][y+1][x] +  A[t%2][y-1][x] + A[t%2][y][x+1]+A[t%2][y][x-1]) + \
									   0.0049 * (A[t%2][y+1][x-1] + A[t%2][y-1][x+1] + A[t%2][y-1][x-1] + A[t%2][y+1][x+1]); 
#define XSLOPE 1
#define YSLOPE 1
#define DATA_TYPE double
#endif

#ifdef CHECK
#define TOLERANCE  0
#endif


int main(int argc, char * argv[]){

	int iam, np;
	double begin, elaspe;
	
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&iam);	
	MPI_Comm_size(MPI_COMM_WORLD,&np);//np=4

	long int t, i, j;
	int NX = atoi(argv[1]);
	int NY = atoi(argv[2]);
	int T  = atoi(argv[3]);
	int ghost_x = atoi(argv[4]);
	int ghost_y = atoi(argv[5]);

	int npx = sqrt(np);
	int npy = npx;
	int ix = ceild(NX,npx);
	int iy = ceild(NY,npy);
	
	int domain_x = ix + 2*ghost_x;
	int domain_y = iy + 2*ghost_y;

	if((iam+1) % npx == 0){ //右边界进程x大小修正
		domain_x = NX - (ix * (npx-1)) + 2*ghost_x;
	}
	if(iam >= npx*(npy-1) && iam < npx*npy){ //上边界进程y大小修正
		domain_y = NY - (iy * (npy-1)) + 2*ghost_y;
	}

	DATA_TYPE (*A)[domain_y][domain_x] = (DATA_TYPE (*)[domain_y][domain_x])malloc(sizeof(DATA_TYPE)*domain_x*domain_y*2);
	if(NULL == A) return 0;

	DATA_TYPE *sbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
	DATA_TYPE *sbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
	DATA_TYPE *rbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
	DATA_TYPE *rbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);

	int count1, count2;
	long traffic = 0;
	long traffic_all = 0;

#ifdef CHECK
	DATA_TYPE (*B)[domain_y][domain_x] = (DATA_TYPE (*)[domain_y][domain_x])malloc(sizeof(DATA_TYPE)*domain_x*domain_y*2);
	if(NULL == B) return 0;

	DATA_TYPE *csbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
	DATA_TYPE *csbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
	DATA_TYPE *crbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
	DATA_TYPE *crbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*domain_x);
#endif

	srand(100);

	for (j = ghost_y; j < domain_y-ghost_y; j++){
		for (i = ghost_x; i < domain_x-ghost_x; i++){
			A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
			//A[0][j][i] = 0;
#if  point == 0
			A[0][j][i] = ((int)A[0][j][i])%2;
#endif
			A[1][j][i] = 0;
#ifdef CHECK
			B[0][j][i] = A[0][j][i];
			B[1][j][i] = 0;
#endif
		}
	}


	if(iam < npx){ //初始化下边界
		for (j = ghost_y-YSLOPE; j < ghost_y; j++){
			for (i = ghost_x; i < domain_x-ghost_x; i++){
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
	}
	if(iam % npx == 0){ //初始化左边界
		for (j = ghost_y; j < domain_y-ghost_y; j++){
			for (i = ghost_x-XSLOPE; i < ghost_x; i++){
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
		if(iam == 0){ //左下角
			for (j = ghost_y-YSLOPE; j < ghost_y; j++){
				for (i = ghost_x-XSLOPE; i < ghost_x; i++){
					A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
					A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
				}
			}
		}
		if(iam == npx*(npy-1)){ //左上角
			for (j = domain_y-ghost_y; j < domain_y-ghost_y+YSLOPE; j++){
				for (i = ghost_x-XSLOPE; i < ghost_x; i++){
					A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
					A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif 
				}
			}
		}
	}
	if((iam+1) % npx == 0){  //初始化右边界
		for (j = ghost_y; j < domain_y-ghost_y; j++){
			for (i = domain_x-ghost_x; i < domain_x-ghost_x+XSLOPE; i++){
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
		if(iam == npx-1){ //右下角
			for (j = ghost_y-YSLOPE; j < ghost_y; j++){
				for (i = domain_x-ghost_x; i < domain_x-ghost_x+XSLOPE; i++){
					A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
					A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
				}
			}
		}
		if(iam == npx*npy-1){ //右上角
			for (j = domain_y-ghost_y; j < domain_y-ghost_y+YSLOPE; j++){
				for (i = domain_x-ghost_x; i < domain_x-ghost_x+XSLOPE; i++){
					A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
					A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
				}
			}
		}
	}
	if(iam >= npx*(npy-1) && iam < npx*npy){ //初始化上边界
		for (j = domain_y-ghost_y; j < domain_y-ghost_y+YSLOPE; j++){
			for (i = ghost_x; i < domain_x-ghost_x; i++){
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
	}

	int level = 0;
	int tt,n;
	int x, y;
	register int ymin, ymax;
	int xmin,xmax;

    MPI_Barrier(MPI_COMM_WORLD);
    if(iam < npx*npy){
        begin = MPI_Wtime();
        for (t = 0; t < T; t++){
            count1 = 0;
            for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){//最下一行，多发送一个下边界，最上层，多发一个上边界
                for(i = ghost_x; i < ghost_x+ghost_x; i++){
                    sbuf1[count1] = A[t%2][j][i];
                    count1++;
                }
            }

            count2 = 0;
            for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){//最下一行，多发送一个下边界，最上层，多发一个上边界
                for(i = domain_x-2*ghost_x; i < domain_x-ghost_x; i++){
                    sbuf2[count2] = A[t%2][j][i];
                    count2++;
                }
            }

            if((iam%npx) & 0x1){//奇数列进程先发送sbuf1，接收sbuf2
                MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 20, MPI_COMM_WORLD);
                traffic += count1;
                MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-1, 21, MPI_COMM_WORLD, &status);
                if((iam+1)%npx != 0){//除每行最右进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 22, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+1, 23, MPI_COMM_WORLD);
                    traffic += count2;
                }
            }
            else{//偶数列进程先接收sbuf1，发送sbuf2
                if((iam+1)%npx != 0){//除每行最右进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 20, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+1, 21, MPI_COMM_WORLD);
                    traffic += count2;
                }
                if(iam%npx != 0){//除每行最左进程
                    MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 22, MPI_COMM_WORLD);
                    traffic += count1;
                    MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-1, 23, MPI_COMM_WORLD, &status);
                }
            }
            if((iam+1)%npx != 0){//除了每行最右进程, 解包rbuf1(横向右侧接收)
                count1 = 0;
                for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){
                    for(i = domain_x-ghost_x; i < domain_x; i++){
                        A[t%2][j][i] = rbuf1[count1];
                        count1++;
                    }
                }
            }
            if(iam%npx != 0){//除了每行最左进程, 解包rbuf2(横向左侧接收)
                count2 = 0;
                for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){
                    for(i = 0; i < ghost_x; i++){
                        A[t%2][j][i] = rbuf2[count2];
                        count2++;
                    }
                }
            }
            if(iam >= npx){//除了最下层进程，打包sbuf1(纵向向下传输)
                count1 = 0;
                for(j = ghost_y; j < ghost_y+ghost_y; j++){
                    for(i = 0; i < domain_x; i++){
                        sbuf1[count1] = A[t%2][j][i];
                        count1++;
                    }
                }
            }
            else count1 = ghost_y*domain_x;

            if(iam < npx*(npy-1)){//除最上层进程，打包sbuf2(纵向向上传输)
                count2 = 0;
                for(j = domain_y-2*ghost_y; j < domain_y-ghost_y; j++){
                    for(i = 0; i < domain_x; i++){
                        sbuf2[count2] = A[t%2][j][i];
                        count2++;
                    }
                }
            }
            else count2 = ghost_y*domain_x;

            if((iam/npx) & 0x1){//奇数行进程先发送csbuf1，接收csbuf2
                MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-npx, 24, MPI_COMM_WORLD);
                traffic += count1;
                MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 25, MPI_COMM_WORLD, &status);
                if(iam < npx*(npy-1)){//除每列最上进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+npx, 26, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 27, MPI_COMM_WORLD);
                    traffic += count2;
                }
            }
            else{//偶数行进程先接收csbuf1，发送csbuf2
                if(iam < npx*(npy-1)){//除每列最上进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+npx, 24, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 25, MPI_COMM_WORLD);
                    traffic += count2;
                }
                if(iam >= npx){//除每列最下进程
                    MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-npx, 26, MPI_COMM_WORLD);
                    traffic += count1;
                    MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 27, MPI_COMM_WORLD, &status);
                }
            }
            if(iam < npx*(npy-1)){//除最上层进程，解包rbuf1(纵向上侧接收)
                count1 = 0;
                for(j = domain_y-ghost_y; j < domain_y; j++){
                    for(i = 0; i < domain_x; i++){
                        A[t%2][j][i] = rbuf1[count1];
                        count1++;
                    }
                }
            }
            if(iam >= npx){//除了最下层进程，解包rbuf2(纵向下侧接收)
                count2 = 0;
                for(j = 0; j < ghost_y; j++){
                    for(i = 0; i < domain_x; i++){
                        A[t%2][j][i] = rbuf2[count2];
                        count2++;
                    }
                }
            }

			for (y = ghost_y; y < domain_y-ghost_y; y++){
				for (x = ghost_x; x < domain_x-ghost_x; x++){
					kernel(A);
				}
			}
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    elaspe = MPI_Wtime()-begin;
	MPI_Reduce(&traffic,&traffic_all,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);

    if(iam == 0)
		printf("NX = %d\tNY = %d\tix = %d\tghost_x = %d\tiy = %d\tghost_y = %d\tMStencil/s = %f\ttraffic = %ld\n",NX,NY,ix,ghost_x,iy,ghost_y,((double)NX * NY * T) / elaspe / 1000000L,traffic_all);

#ifdef CHECK
    if(iam < npx*npy){
        for (t = 0; t < T; t++){
            count1 = 0;
            for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){//最下一行，多发送一个下边界，最上层，多发一个上边界
                for(i = ghost_x; i < ghost_x+XSLOPE; i++){
                    sbuf1[count1] = B[t%2][j][i];
                    count1++;
                }
            }

            count2 = 0;
            for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){//最下一行，多发送一个下边界，最上层，多发一个上边界
                for(i = domain_x-ghost_x-XSLOPE; i < domain_x-ghost_x; i++){
                    sbuf2[count2] = B[t%2][j][i];
                    count2++;
                }
            }

            if((iam%npx) & 0x1){//奇数列进程先发送sbuf1，接收sbuf2
                MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 20, MPI_COMM_WORLD);
                MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-1, 21, MPI_COMM_WORLD, &status);
                if((iam+1)%npx != 0){//除每行最右进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 22, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+1, 23, MPI_COMM_WORLD);
                }
            }
            else{//偶数列进程先接收sbuf1，发送sbuf2
                if((iam+1)%npx != 0){//除每行最右进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 20, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+1, 21, MPI_COMM_WORLD);
                }
                if(iam%npx != 0){//除每行最左进程
                    MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 22, MPI_COMM_WORLD);
                    MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-1, 23, MPI_COMM_WORLD, &status);
                }
            }
            if((iam+1)%npx != 0){//除了每行最右进程, 解包rbuf1(横向右侧接收)
                count1 = 0;
                for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){
                    for(i = domain_x-ghost_x; i < domain_x-ghost_x+XSLOPE; i++){
                        B[t%2][j][i] = rbuf1[count1];
                        count1++;
                    }
                }
            }
            if(iam%npx != 0){//除了每行最左进程, 解包rbuf2(横向左侧接收)
                count2 = 0;
                for(j = ((iam<npx)? ghost_y-YSLOPE : ghost_y); j < ((iam<npx*(npy-1)) ? domain_y-ghost_y : domain_y-ghost_y+YSLOPE); j++){
                    for(i = ghost_x-XSLOPE; i < ghost_x; i++){
                        B[t%2][j][i] = rbuf2[count2];
                        count2++;
                    }
                }
            }
            if(iam >= npx){//除了最下层进程，打包sbuf1(纵向向下传输)
                count1 = 0;
                for(j = ghost_y; j < ghost_y+YSLOPE; j++){
                    for(i = 0; i < domain_x; i++){
                        sbuf1[count1] = B[t%2][j][i];
                        count1++;
                    }
                }
            }
            else count1 = YSLOPE*domain_x;

            if(iam < npx*(npy-1)){//除最上层进程，打包sbuf2(纵向向上传输)
                count2 = 0;
                for(j = domain_y-ghost_y-YSLOPE; j < domain_y-ghost_y; j++){
                    for(i = 0; i < domain_x; i++){
                        sbuf2[count2] = B[t%2][j][i];
                        count2++;
                    }
                }
            }
            else count2 = YSLOPE*domain_x;

            if((iam/npx) & 0x1){//奇数行进程先发送csbuf1，接收csbuf2
                MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-npx, 24, MPI_COMM_WORLD);
                MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 25, MPI_COMM_WORLD, &status);
                if(iam < npx*(npy-1)){//除每列最上进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+npx, 26, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 27, MPI_COMM_WORLD);
                }
            }
            else{//偶数行进程先接收csbuf1，发送csbuf2
                if(iam < npx*(npy-1)){//除每列最上进程
                    MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+npx, 24, MPI_COMM_WORLD, &status);
                    MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 25, MPI_COMM_WORLD);
                }
                if(iam >= npx){//除每列最下进程
                    MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-npx, 26, MPI_COMM_WORLD);
                    MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 27, MPI_COMM_WORLD, &status);
                }
            }
            if(iam < npx*(npy-1)){//除最上层进程，解包rbuf1(纵向上侧接收)
                count1 = 0;
                for(j = domain_y-ghost_y; j < domain_y-ghost_y+YSLOPE; j++){
                    for(i = 0; i < domain_x; i++){
                        B[t%2][j][i] = rbuf1[count1];
                        count1++;
                    }
                }
            }
            if(iam >= npx){//除了最下层进程，解包rbuf2(纵向下侧接收)
                count2 = 0;
                for(j = ghost_y-YSLOPE; j < ghost_y; j++){
                    for(i = 0; i < domain_x; i++){
                        B[t%2][j][i] = rbuf2[count2];
                        count2++;
                    }
                }
            }

			for (y = ghost_y; y < domain_y-ghost_y; y++){
				for (x = ghost_x; x < domain_x-ghost_x; x++){
					kernel(B);
				}
			}
        }
    }
	for (j = ghost_y; j < domain_y-ghost_y; j++){
		for (i = ghost_x; i < domain_x-ghost_x; i++){
            if(myabs(A[T%2][j][i],B[T%2][j][i]) > TOLERANCE)
				printf("%d:\tNaive[%d][%d] = %f, Check = %f: FAILED!\n", iam, j, i, B[T%2][j][i], A[T%2][j][i]);
		}
	}
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}



