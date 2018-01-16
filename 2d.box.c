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
#define point 5
#endif

#if point == 5
#define  kernel(A)  A[(t+1)%2][y][x] = 0.125 * (A[t%2][y+1][x] - 2.0 * A[t%2][y][x] + A[t%2][y-1][x]) + \
									   0.125 * (A[t%2][y][x+1] - 2.0 * A[t%2][y][x] + A[t%2][y][x-1]) + \
									   A[t%2][y][x];
//#define  kernel(A)  A[(t+1)%2][y][x] = A[t%2][y][x] + 1;
#define XSLOPE 1
#define YSLOPE 1
#define DATA_TYPE double
#elif point == 9
#define  kernel(A) A[(t+1)%2][x][y] =  0.96 * A[t%2][x][y] + \
									   0.0051 * (A[t%2][x+1][y] +  A[t%2][x-1][y] + A[t%2][x][y+1]+A[t%2][x][y-1]) + \
									   0.0049 * (A[t%2][x+1][y-1] + A[t%2][x-1][y+1] + A[t%2][x-1][y-1] + A[t%2][x+1][y+1]); 
#define XSLOPE 1
#define YSLOPE 1
#define DATA_TYPE double
#elif point == 0
#define kernel(A)  A[(t+1)%2][x][y] = b2s23(A[t%2][x][y], A[t%2][x-1][y+1] + A[t%2][x-1][y] + \
														  A[t%2][x-1][y-1] + A[t%2][x][y+1] + \
														  A[t%2][x][y-1]   + A[t%2][x+1][y+1] + \
														  A[t%2][x+1][y]   + A[t%2][x+1][y-1]);
#define XSLOPE 1
#define YSLOPE 1
#define DATA_TYPE int
int b2s23(int cell, int neighbors) {
  if((cell == 1 && ((neighbors < 2) || (neighbors > 3)))) {
    return 0;
  }
  if((cell == 1 && (neighbors == 2 || neighbors == 3))) {
    return 1;
  }
  if((cell == 0 && neighbors == 3)) {
    return 1;
  }
  return cell;
}
#endif


#ifdef CHECK
#define TOLERANCE  0.000001
#endif


int main(int argc, char * argv[]) {

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
	int bx = atoi(argv[4]);
	int by = atoi(argv[5]);

	int npx = sqrt(np);
	int npy = npx;
	int ix = ceild(NX,npx);
	int iy = ceild(NY,npy);
	int Bx = ix - bx;
	int By = iy - by;
	int tb = min(((Bx-bx)/2)/XSLOPE,((By-by)/2)/YSLOPE);
	Bx = (ix + 2*tb*XSLOPE)/2;
	By = (iy + 2*tb*YSLOPE)/2;
	bx = ix - Bx;
	by = iy - By;

	int domain_x = ix + tb + 2*XSLOPE;
	int domain_y = iy + tb + 2*YSLOPE;

	if((iam+1) % npx == 0) { //右边界进程x大小修正
		domain_x = NX - (ix * (npx-1)) + 2*XSLOPE;
	}
	if(iam >= npx*(npy-1) && iam < npx*npy) { //上边界进程y大小修正
		domain_y = NY - (iy * (npy-1)) + 2*YSLOPE;
	}

	DATA_TYPE (*A)[domain_y][domain_x] = (DATA_TYPE (*)[domain_y][domain_x])malloc(sizeof(DATA_TYPE)*domain_x*domain_y*2);
	if(NULL == A) return 0;
	
	DATA_TYPE *sbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*(By+tb+YSLOPE)*(tb+XSLOPE)*2);//t=0,t=1
	DATA_TYPE *sbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*(Bx+tb+XSLOPE)*(tb+YSLOPE)*2);
	DATA_TYPE *rbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*(By+tb+YSLOPE)*(tb+XSLOPE)*2);
	DATA_TYPE *rbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*(Bx+tb+XSLOPE)*(tb+YSLOPE)*2);
	int count1, count2;

#ifdef CHECK
	DATA_TYPE (*B)[domain_y][domain_x] = (DATA_TYPE (*)[domain_y][domain_x])malloc(sizeof(DATA_TYPE)*domain_x*domain_y*2);
	if(NULL == B) return 0;

	DATA_TYPE *csbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*(tb+XSLOPE)*2);//向左发送
	DATA_TYPE *csbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*XSLOPE*2);//向右发送
	DATA_TYPE *csbuf3 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_x*(tb+YSLOPE)*2);//向下发送
	DATA_TYPE *csbuf4 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_x*YSLOPE*2);//向上发送
	DATA_TYPE *crbuf1 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*(tb+XSLOPE)*2);//从右面接收
	DATA_TYPE *crbuf2 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_y*XSLOPE*2);//从左面接收
	DATA_TYPE *crbuf3 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_x*(tb+YSLOPE)*2);//从上面接收
	DATA_TYPE *crbuf4 = (DATA_TYPE *)malloc(sizeof(DATA_TYPE)*domain_x*YSLOPE*2);//从下面接收
#endif

	srand(100);

	for (j = YSLOPE; j < min(iy+YSLOPE, domain_y-YSLOPE); j++) { //初始化内部B0和B1起始区域
		for (i = XSLOPE; i < min(ix+XSLOPE, domain_x-XSLOPE); i++) {
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

	if(iam < npx) { //初始化下边界
		for (j = 0; j < YSLOPE; j++) {
			for (i = XSLOPE; i < min(ix+XSLOPE, domain_x-XSLOPE); i++) {
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
	}
	if(iam % npx == 0) { //初始化左边界
		for (j = YSLOPE; j < min(iy+YSLOPE, domain_y-YSLOPE); j++) {
			for (i = 0; i < XSLOPE; i++) {
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
		if(iam == 0) { //左下角
			for (j = 0; j < YSLOPE; j++) {
				for (i = 0; i < XSLOPE; i++) {
					A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
					A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
				}
			}
		}
		if(iam == npx*(npy-1)) { //左上角
			for (j = domain_y-YSLOPE; j < domain_y; j++) {
				for (i = 0; i < XSLOPE; i++) {
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
	if((iam+1) % npx == 0) {  //初始化右边界
		for (j = YSLOPE; j < min(iy+YSLOPE, domain_y-YSLOPE); j++) {
			for (i = domain_x-XSLOPE; i < domain_x; i++) {
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
		if(iam == npx-1) { //右下角
			for (j = 0; j < YSLOPE; j++) {
				for (i = domain_x-XSLOPE; i < domain_x; i++) {
					A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
					A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
				}
			}
		}
		if(iam == npx*npy-1) { //右上角
			for (j = domain_y-YSLOPE; j < domain_y; j++) {
				for (i = domain_x-XSLOPE; i < domain_x; i++) {
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
	if(iam >= npx*(npy-1) && iam < npx*npy) { //初始化上边界
		for (j = domain_y-YSLOPE; j < domain_y; j++) {
			for (i = XSLOPE; i < min(ix+XSLOPE, domain_x-XSLOPE); i++) {
				A[0][j][i] = (DATA_TYPE) (1.0 * (rand() % 1024));
				A[1][j][i] = 0;
#ifdef CHECK
				B[0][j][i] = A[0][j][i];
				B[1][j][i] = 0;
#endif
			}
		}
	}

    int xleft02[2] = {XSLOPE, XSLOPE+ix/2};
	int ybottom02[2] = {YSLOPE, YSLOPE+ix/2};
	int xleft11[2] = {XSLOPE+Bx, XSLOPE+(Bx-bx)/2};//垂直开始,初始左边界
	int ybottom11[2] = {YSLOPE, YSLOPE+iy/2};//初始下边界
	int xleft12[2] = {XSLOPE, XSLOPE+ix/2};//水平开始,初始左边界
	int ybottom12[2] = {YSLOPE+By, YSLOPE+(By-by)/2};//初始下边界
	
	int nb02[2] = {1, 1};
	int nb1[2] = {1, 1};
	int xnb02[2] = {1, 1};
	int xnb1[2] = {1, 1};

	if(iam == 0) {
		nb1[0] = 2;
		nb1[1] = 2;
		nb02[1] = 4;
		xnb02[1] = 2;
		xnb1[0] = 2;
	}
	else if(iam < npx) {//下边界
		nb1[1] = 2;
		nb02[1] = 2;
	}
	else if(iam%npx == 0) {//左边界
		nb1[0] = 2;
		nb02[1] = 2;
		xnb02[1] = 2;
		xnb1[0] = 2;
	}

	int level = 0;
	int tt,n;
	int x, y;
	register int ymin, ymax;
	int xmin,xmax;

    MPI_Barrier(MPI_COMM_WORLD);

    begin = MPI_Wtime();

	for(tt =- tb; tt < T; tt += tb){
		//if(iam==0) printf("before b0b2, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);
		for(n = 0; n < nb02[level]; n++) {
			for(t = max(tt,0); t < min(tt + 2*tb, T); t++) { 

				xmin = max(         XSLOPE,   xleft02[level] - (n%xnb02[level])*ix      + myabs(t+1,tt+tb) * XSLOPE);
				xmax = min(domain_x-XSLOPE,   xleft02[level] - (n%xnb02[level])*ix + Bx - myabs(t+1,tt+tb) * XSLOPE);
				ymin = max(         YSLOPE, ybottom02[level] - (n/xnb02[level])*iy      + myabs(t+1,tt+tb) * YSLOPE);
				ymax = min(domain_y-YSLOPE, ybottom02[level] - (n/xnb02[level])*iy + By - myabs(t+1,tt+tb) * YSLOPE);

				for(y = ymin; y < ymax; y++) {
#pragma simd
					for(x = xmin; x < xmax; x++) {
						kernel(A);
						//if(iam==2) printf("%d\tt=%d\tB0B2\t(%d,%d)\n",iam,t,x,y);
					}
				}
			}
		}
		//if(iam==0) printf("after b0b2, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);
		if(level == 0) {
			if(iam%npx != 0) {//除了每行最左进程, 打包sbuf1(横向向左传输)
				count1 = 0;
				for(t = 0; t < 2; t++) {
					for(j = ((iam<npx)? 0 : YSLOPE); j < YSLOPE+By; j++) {//最下一行，多发送一个下边界
						for(i = XSLOPE; i < XSLOPE+tb+XSLOPE; i++) {
							sbuf1[count1] = A[t][j][i];
							count1++;
						}
					}
				}
			}
			else count1 = ((iam<npx) ? ((By+YSLOPE)*(tb+XSLOPE)*2) : (By*(tb+XSLOPE)*2));
			
			if(iam >= npx) {//除了最下层进程，打包sbuf2(纵向向下传输)
				count2 = 0;
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE; j < YSLOPE+tb+YSLOPE; j++) {
						for(i = ((iam%npx==0)? 0 : XSLOPE); i < XSLOPE+Bx; i++) {//最左一列，多发送一个左边界
							sbuf2[count2] = A[t][j][i];
							count2++;
						}
					}
				}
			}
			else count2 = ((iam%npx==0) ? ((tb+YSLOPE)*(Bx+XSLOPE)*2) : ((tb+YSLOPE)*Bx*2));

			if((iam%npx) & 0x1) {//奇数列进程先发送sbuf1
				MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 0, MPI_COMM_WORLD);
				if((iam+1)%npx != 0) {//除每行最右进程
					MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 1, MPI_COMM_WORLD, &status);
				}
			}
			else {//偶数列进程先接收sbuf14
				if((iam+1)%npx != 0) {//除每行最右进程
					MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 0, MPI_COMM_WORLD, &status);
				}
				if(iam%npx != 0) {//除每行最左进程
					MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 1, MPI_COMM_WORLD);
				}
			}
			if((iam/npx) & 0x1) {//奇数行进程先发送sbuf2
				MPI_Send(sbuf2, count2, MPI_DOUBLE, iam-npx, 2, MPI_COMM_WORLD);
				if(iam < npx*(npy-1)) {//除每列最上进程
					MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam+npx, 3, MPI_COMM_WORLD, &status);
				}
			}
			else {//偶数行进程先接收sbuf2
				if(iam < npx*(npy-1)) {//除每列最上进程
					MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam+npx, 2, MPI_COMM_WORLD, &status);
				}
				if(iam >= npx) {//除每列最下进程
					MPI_Send(sbuf2, count2, MPI_DOUBLE, iam-npx, 3, MPI_COMM_WORLD);
				}
			}

			if((iam+1)%npx != 0) {//除了每行最右进程, 解包rbuf1(横向右侧接收)
				count1 = 0;
				for(t = 0; t < 2; t++) {
					for(j = ((iam<npx)? 0 : YSLOPE); j < YSLOPE+By; j++) {
						for(i = XSLOPE+ix; i < XSLOPE+ix+tb+XSLOPE; i++) {
							A[t][j][i] = rbuf1[count1];
							count1++;
							//if(iam==2) printf("%d\tt=%d\trB0B2\t(%d,%d)\n",iam,t,i,j);
						}
					}
				}
			}
					
			if(iam < npx*(npy-1)) {//除了最上层进程，解包rbuf2(纵向上侧接收)
				count2 = 0;
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE+iy; j < YSLOPE+iy+tb+YSLOPE; j++) {
						for(i = ((iam%npx==0)? 0 : XSLOPE); i < XSLOPE+Bx; i++) {
							A[t][j][i] = rbuf2[count2];
							count2++;
						}
					}
				}
			}
			}
		else {//level==1
			if((iam+1)%npx != 0) {//除了每行最右进程, 打包sbuf1(横向向右传输)
				count1 = 0;
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE+By-tb; j < min(domain_y, YSLOPE+2*By-tb); j++) {
						for(i = XSLOPE+Bx; i < XSLOPE+Bx+tb+XSLOPE; i++) {
							sbuf1[count1] = A[t][j][i];
							count1++;
						}
					}
				}
				if(iam < npx) {//最下层进程要多传输下边界的B0
					for(t = 0; t < 2; t++) {
						for(j = YSLOPE; j < YSLOPE+tb; j++) {
							for(i = XSLOPE+Bx; i < XSLOPE+Bx+tb+XSLOPE; i++) {
								sbuf1[count1] = A[t][j][i];
								//if(iam == 1) printf("%d:\tA[%d][%d][%d] == %f\n",iam,t,j,i,A[t][j][i]);
								count1++;
							}
						}
					}
				}
				if((iam+2)%npx == 0) {
					MPI_Send(&count1, 1, MPI_INT, iam+1, 10, MPI_COMM_WORLD);
				}
			}
			else {
				MPI_Recv(&count1, 1, MPI_INT, iam-1, 10, MPI_COMM_WORLD, &status);
			}
				
			if(iam < npx*(npy-1)) {//除了最上层进程，打包sbuf2(纵向向上传输)
				count2 = 0;
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE+By; j < YSLOPE+By+tb+YSLOPE; j++) {
						for(i = XSLOPE+Bx-tb; i < min(domain_x, XSLOPE+2*Bx-tb); i++) {
							sbuf2[count2] = A[t][j][i];
							count2++;
						}
					}
				}
				if(iam%npx == 0) {//最左侧进程要多传输左边界的B0
					for(t = 0; t < 2; t++) {
						for(j = YSLOPE+By; j < YSLOPE+By+tb+YSLOPE; j++) {
							for(i = XSLOPE; i < XSLOPE+tb; i++) {
								sbuf2[count2] = A[t][j][i];
								count2++;
							}
						}
					}
				}
				if(iam >= npx*(npy-2)) {
					MPI_Send(&count2, 1, MPI_INT, iam+npx, 11, MPI_COMM_WORLD);
				}
			}
			else {
				MPI_Recv(&count2, 1, MPI_INT, iam-npx, 11, MPI_COMM_WORLD, &status);
			}

			if((iam%npx) & 0x1) {//奇数列进程先发送sbuf1
				if((iam+1)%npx != 0) {//除每行最右进程
					MPI_Send(sbuf1, count1, MPI_DOUBLE, iam+1, 0, MPI_COMM_WORLD);
				}
				MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam-1, 1, MPI_COMM_WORLD, &status);
				//printf("%d:\t level=%d \t B0 rbuf1 recv!\n",iam,level);
			}
			else {//偶数列进程先接收sbuf1
				if(iam%npx != 0) {//除每行最左进程
					MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam-1, 0, MPI_COMM_WORLD, &status);
				}
				if((iam+1)%npx != 0) {//除每行最右进程
					MPI_Send(sbuf1, count1, MPI_DOUBLE, iam+1, 1, MPI_COMM_WORLD);
					//printf("%d:\t level=%d \t B0 rbuf1 send!\n",iam,level);
				}
			}
			if((iam/npx) & 0x1) {//奇数行进程先发送sbuf2
				if(iam < npx*(npy-1)) {//除每列最上进程
					MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 2, MPI_COMM_WORLD);
				}
				MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 3, MPI_COMM_WORLD, &status);
				//printf("%d:\t level=%d \t B0 rbuf2 recv!\n",iam,level);
			}
			else {//偶数行进程先接收sbuf2
				if(iam >= npx) {//除每列最下进程
					MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 2, MPI_COMM_WORLD, &status);
				}
				if(iam < npx*(npy-1)) {//除每列最上进程
					MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 3, MPI_COMM_WORLD);
					//printf("%d:\t level=%d \t B0 rbuf2 send!\n",iam,level);
				}
			}

			if(iam%npx != 0) {//除了每行最左进程, 解包rbuf1(横向左侧接收)
				count1 = 0;
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE+By-tb; j < min(domain_y, YSLOPE+2*By-tb); j++) {
						for(i = 0; i < tb+XSLOPE; i++) {
							A[t][j][i] = rbuf1[count1];
							count1++;
						}
					}
				}
				if(iam < npx) {//最下层进程要多解包下边界的B0
					for(t = 0; t < 2; t++) {
						for(j = YSLOPE; j < YSLOPE+tb; j++) {
							for(i = 0; i < tb+XSLOPE; i++) {
								A[t][j][i] = rbuf1[count1];
								//if(iam == 1) printf("%d:\tA[%d][%d][%d] == %f\n",iam,t,j,i,A[t][j][i]);
								count1++;
							}
						}
					}
				}
			}
					
			if(iam >= npx) {//除了最下层进程，解包rbuf2(纵向下侧接收)
				count2 = 0;
				for(t = 0; t < 2; t++) {
					for(j = 0; j < tb+YSLOPE; j++) {
						for(i = XSLOPE+Bx-tb; i < min(domain_x, XSLOPE+2*Bx-tb); i++) {
							A[t][j][i] = rbuf2[count2];
							count2++;
						}
					}
				}
				if(iam%npx == 0) {//最左侧进程要多解包左边界的B0
					for(t = 0; t < 2; t++) {
						for(j = 0; j < tb+YSLOPE; j++) {
							for(i = XSLOPE; i < XSLOPE+tb; i++) {
								A[t][j][i] = rbuf2[count2];
								count2++;
							}
						}
					}
				}
			}
		}
		//if(iam==0) printf("before b1, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);

		for(n = 0; n < nb1[0] + nb1[1]; n++) {

			for(t = tt + tb; t < min(tt + 2*tb, T); t++) {
				if(n < nb1[level]) {
					xmin = max(     XSLOPE,       xleft11[level] - (n%xnb1[level]) * ix       - (t+1-tt-tb) * XSLOPE);
					xmax = min(domain_x-XSLOPE,   xleft11[level] - (n%xnb1[level]) * ix  + bx + (t+1-tt-tb) * XSLOPE);
					ymin = max(     YSLOPE,     ybottom11[level] - (n/xnb1[level]) * iy       + (t+1-tt-tb) * YSLOPE);
					ymax = min(domain_y-YSLOPE, ybottom11[level] - (n/xnb1[level]) * iy  + By - (t+1-tt-tb) * YSLOPE);
				}
				else {
					xmin = max(     XSLOPE,       xleft12[level] - ((n-nb1[level])%xnb1[1-level]) * ix      + (t+1-tt-tb) * XSLOPE);
					xmax = min(domain_x-XSLOPE,   xleft12[level] - ((n-nb1[level])%xnb1[1-level]) * ix + Bx - (t+1-tt-tb) * XSLOPE);
					ymin = max(     YSLOPE,     ybottom12[level] - ((n-nb1[level])/xnb1[1-level]) * iy      - (t+1-tt-tb) * YSLOPE);
					ymax = min(domain_y-YSLOPE, ybottom12[level] - ((n-nb1[level])/xnb1[1-level]) * iy + by + (t+1-tt-tb) * YSLOPE);
				}
				for(y = ymin; y < ymax; y++) {
#pragma simd
					for(x = xmin; x < xmax; x++) {
						//if(iam==2) printf("%d\tt=%d\tB1\t(%d,%d)\n",iam,t,x,y);
						kernel(A);
					}
				}
			}
		}
		//if(iam==0) printf("after b1, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);
		//printf("%d:\t level=%d \t B1 completed!\n",iam,level);
		if(level == 0) {
		if(iam%npx != 0) {//除了每行最左进程, 打包sbuf1(横向向左传输)
			count1 = 0;
			for(t = 0; t < 2; t++) {
				for(i = XSLOPE; i < XSLOPE+tb+XSLOPE; i++) {
					for(j = By-(i-XSLOPE); j < min(domain_y, iy+YSLOPE+(i-XSLOPE)+YSLOPE); j++) {
						sbuf1[count1] = A[t][j][i];
						count1++;
					}
				}
			}
			if(iam < npx) {//最下层进程要多传输下边界的B1
				for(t = 0; t < 2; t++) {
					for(i = XSLOPE; i < XSLOPE+tb+XSLOPE; i++) {
						for(j = YSLOPE; j < YSLOPE+(i-XSLOPE)+YSLOPE; j++) {
							sbuf1[count1] = A[t][j][i];
							//if(iam == 1) printf("%d:\tA[%d][%d][%d] == %f\n",iam,t,j,i,A[t][j][i]);
							count1++;
						}
					}
				}
			}
			if((iam-1)%npx == 0) {
				MPI_Send(&count1, 1, MPI_INT, iam-1, 4, MPI_COMM_WORLD);
			}
		}
		else {
			MPI_Recv(&count1, 1, MPI_INT, iam+1, 4, MPI_COMM_WORLD, &status);
		}
			
		if(iam >= npx) {//除了最下层进程，打包sbuf2(纵向向下传输)
			count2 = 0;
			for(t = 0; t < 2; t++) {
				for(j = YSLOPE; j < YSLOPE+tb+YSLOPE; j++) {
					for(i = Bx-(j-YSLOPE); i < min(domain_x, ix+XSLOPE+(j-YSLOPE)+XSLOPE); i++) {
						sbuf2[count2] = A[t][j][i];
						count2++;
					}
				}
			}
			if(iam%npx == 0) {//最左侧进程要多传输左边界的B1
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE; j < YSLOPE+tb+YSLOPE; j++) {
						for(i = XSLOPE; i < XSLOPE+(j-YSLOPE)+XSLOPE; i++) {
							sbuf2[count2] = A[t][j][i];
							count2++;
						}
					}
				}
			}
			if(iam < 2*npx) {
				MPI_Send(&count2, 1, MPI_INT, iam-npx, 5, MPI_COMM_WORLD);
			}
		}
		else {
			MPI_Recv(&count2, 1, MPI_INT, iam+npx, 5, MPI_COMM_WORLD, &status);
		}

		if((iam%npx) & 0x1) {//奇数列进程先发送sbuf1
			MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 6, MPI_COMM_WORLD);
			if((iam+1)%npx != 0) {//除每行最右进程
				MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 7, MPI_COMM_WORLD, &status);
			}
		}
		else {//偶数列进程先接收sbuf1
			if((iam+1)%npx != 0) {//除每行最右进程
				MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam+1, 6, MPI_COMM_WORLD, &status);
			}
			if(iam%npx != 0) {//除每行最左进程
				MPI_Send(sbuf1, count1, MPI_DOUBLE, iam-1, 7, MPI_COMM_WORLD);
			}
		}
		if((iam/npx) & 0x1) {//奇数行进程先发送sbuf2
			MPI_Send(sbuf2, count2, MPI_DOUBLE, iam-npx, 8, MPI_COMM_WORLD);
			if(iam < npx*(npy-1)) {//除每列最上进程
				MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam+npx, 9, MPI_COMM_WORLD, &status);
			}
		}
		else {//偶数行进程先接收sbuf2
			if(iam < npx*(npy-1)) {//除每列最上进程
				MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam+npx, 8, MPI_COMM_WORLD, &status);
			}
			if(iam >= npx) {//除每列最下进程
				MPI_Send(sbuf2, count2, MPI_DOUBLE, iam-npx, 9, MPI_COMM_WORLD);
			}
		}

		if((iam+1)%npx != 0) {//除了每行最右进程, 解包rbuf1(横向右侧接收)
			count1 = 0;
			for(t = 0; t < 2; t++) {
				for(i = XSLOPE+ix; i < XSLOPE+ix+tb+XSLOPE; i++) {
					for(j = By-(i-XSLOPE-ix); j < min(domain_y, iy+YSLOPE+(i-XSLOPE-ix)+YSLOPE); j++) {
						A[t][j][i] = rbuf1[count1];
						count1++;
						//if(iam==2) printf("%d\tt=%d\trB1\t(%d,%d)\n",iam,t,i,j);
					}
				}
			}
			if(iam < npx) {//最下层进程要多解包下边界的B1
				for(t = 0; t < 2; t++) {
					for(i = XSLOPE+ix; i < XSLOPE+ix+tb+XSLOPE; i++) {
						for(j = YSLOPE; j < YSLOPE+(i-XSLOPE-ix)+YSLOPE; j++) {
							A[t][j][i] = rbuf1[count1];
							//if(iam == 0) printf("%d:\tA[%d][%d][%d] == %f\n",iam,t,j,i,A[t][j][i]);
							count1++;
						}
					}
				}
			}
		}
				
		if(iam < npx*(npy-1)) {//除了最上层进程，解包rbuf2(纵向上侧接收)
			count2 = 0;
			for(t = 0; t < 2; t++) {
				for(j = YSLOPE+iy; j < YSLOPE+iy+tb+YSLOPE; j++) {
					for(i = Bx-(j-YSLOPE-iy); i < min(domain_x, ix+XSLOPE+(j-YSLOPE-iy)+XSLOPE); i++) {
						A[t][j][i] = rbuf2[count2];
						count2++;
					}
				}
			}
			if(iam%npx == 0) {//最左侧进程要多解包左边界的B1
				for(t = 0; t < 2; t++) {
					for(j = YSLOPE+iy; j < YSLOPE+iy+tb+YSLOPE; j++) {
						for(i = XSLOPE; i < XSLOPE+(j-YSLOPE-iy)+XSLOPE; i++) {
							A[t][j][i] = rbuf2[count2];
							count2++;
						}
					}
				}
			}
		}
		}
		else { //level == 1
		if((iam+1)%npx != 0) {//除了每行最右进程, 打包sbuf1(横向向右传输)
			count1 = 0;
			for(t = 0; t < 2; t++) {
				for(i = ix; i < ix+tb+XSLOPE; i++) {
					for(j = i-ix; j < min(domain_y, YSLOPE+By-(i-ix)+YSLOPE); j++) {
						sbuf1[count1] = A[t][j][i];
						count1++;
					}
				}
			}
			if((iam+2)%npx == 0) {
				MPI_Send(&count1, 1, MPI_INT, iam+1, 4, MPI_COMM_WORLD);
			}
		}
		else {
			MPI_Recv(&count1, 1, MPI_INT, iam-1, 4, MPI_COMM_WORLD, &status);
		}
			
		if(iam < npx*(npy-1)) {//除了最上层进程，打包sbuf2(纵向向上传输)
			count2 = 0;
			for(t = 0; t < 2; t++) {
				for(j = YSLOPE+By; j < YSLOPE+By+tb+YSLOPE; j++) {
					for(i = j-YSLOPE-By; i < min(domain_x, XSLOPE+Bx-(j-YSLOPE-By)+XSLOPE); i++) {
						sbuf2[count2] = A[t][j][i];
						count2++;
					}
				}
			}
			if(iam >= npx*(npy-2)) {
				MPI_Send(&count2, 1, MPI_INT, iam+npx, 5, MPI_COMM_WORLD);
			}
		}
		else {
			MPI_Recv(&count2, 1, MPI_INT, iam-npx, 5, MPI_COMM_WORLD, &status);
		}

		if((iam%npx) & 0x1) {//奇数列进程先发送sbuf1
			if((iam+1)%npx != 0) {//除每行最右进程
				MPI_Send(sbuf1, count1, MPI_DOUBLE, iam+1, 6, MPI_COMM_WORLD);
			}
			MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam-1, 7, MPI_COMM_WORLD, &status);
		}
		else {//偶数列进程先接收sbuf1
			if(iam%npx != 0) {//除每行最左进程
				MPI_Recv(rbuf1, count1, MPI_DOUBLE, iam-1, 6, MPI_COMM_WORLD, &status);
			}
			if((iam+1)%npx != 0) {//除每行最右进程
				MPI_Send(sbuf1, count1, MPI_DOUBLE, iam+1, 7, MPI_COMM_WORLD);
			}
		}
		if((iam/npx) & 0x1) {//奇数行进程先发送sbuf2
			if(iam < npx*(npy-1)) {//除每列最上进程
				MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 8, MPI_COMM_WORLD);
			}
			MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 9, MPI_COMM_WORLD, &status);
		}
		else {//偶数行进程先接收sbuf2
			if(iam >= npx) {//除每列最下进程
				MPI_Recv(rbuf2, count2, MPI_DOUBLE, iam-npx, 8, MPI_COMM_WORLD, &status);
			}
			if(iam < npx*(npy-1)) {//除每列最上进程
				MPI_Send(sbuf2, count2, MPI_DOUBLE, iam+npx, 9, MPI_COMM_WORLD);
			}
		}

		if(iam%npx != 0) {//除了每行最左进程, 解包rbuf1(横向左侧接收)
			count1 = 0;
			for(t = 0; t < 2; t++) {
				for(i = 0; i < tb+XSLOPE; i++) {
					for(j = i; j < min(domain_y, YSLOPE+By-i+YSLOPE); j++) {
						A[t][j][i] = rbuf1[count1];
						count1++;
					}
				}
			}
		}
				
		if(iam >= npx) {//除了最下层进程，解包rbuf2(纵向下侧接收)
			count2 = 0;
			for(t = 0; t < 2; t++) {
				for(j = 0; j < tb+YSLOPE; j++) {
					for(i = j; i < min(domain_x, XSLOPE+Bx-j+XSLOPE); i++) {
						A[t][j][i] = rbuf2[count2];
						count2++;
					}
				}
			}
		}
		}


		level = 1- level;
	}

    MPI_Barrier(MPI_COMM_WORLD);
    elaspe = MPI_Wtime()-begin;

    if(iam == 0)
		printf("level = %d\tN = %d\tBx = %d\tbx = %d\ttb = %d\tMStencil/s = %f\n",level,NX,Bx,bx,tb,((double)NX * NY * T) / elaspe / 1000000L);

#ifdef CHECK
		//if(iam==0) printf("before check, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);

//	if(iam%npx != 0) {//除了每行最左进程, 打包csbuf1(横向向左传输)
		count1 = 0;
		for(t = 0; t < 2; t++) {
			for(j = ((iam<npx)? 0 : YSLOPE); j < ((iam<npx*(npy-1)) ? (YSLOPE+iy) : domain_y); j++) {//最下一行，多发送一个下边界，最上层，多发一个上边界
				for(i = XSLOPE; i < XSLOPE+tb+XSLOPE; i++) {
					csbuf1[count1] = B[t][j][i];
					count1++;
				}
			}
		}
//	}
//	else count1 = ((iam<npx) ? ((iy+YSLOPE)*(tb+XSLOPE)*2) : (iy*(tb+XSLOPE)*2));

//	if((iam+1)%npx != 0) {//除每行最右进程, 打包csbuf2(横向向右传输)
		count2 = 0;
		for(t = 0; t < 2; t++) {
			for(j = ((iam<npx)? 0 : YSLOPE); j < ((iam<npx*(npy-1)) ? (YSLOPE+iy) : domain_y); j++) {//最下一行，多发送一个下边界，最上层，多发一个上边界
				for(i = ix; i < ix+XSLOPE; i++) {
					csbuf2[count2] = B[t][j][i];
					count2++;
				}
			}
		}
//	}
//	else count2 = ((iam<npx) ? ((iy+YSLOPE)*XSLOPE*2) : (iy*XSLOPE*2));

	if((iam%npx) & 0x1) {//奇数列进程先发送csbuf1，接收csbuf2
		MPI_Send(csbuf1, count1, MPI_DOUBLE, iam-1, 20, MPI_COMM_WORLD);
		MPI_Recv(crbuf2, count2, MPI_DOUBLE, iam-1, 21, MPI_COMM_WORLD, &status);
		if((iam+1)%npx != 0) {//除每行最右进程
			MPI_Recv(crbuf1, count1, MPI_DOUBLE, iam+1, 22, MPI_COMM_WORLD, &status);
			MPI_Send(csbuf2, count2, MPI_DOUBLE, iam+1, 23, MPI_COMM_WORLD);
		}
	}
	else {//偶数列进程先接收csbuf1，发送csbuf2
		if((iam+1)%npx != 0) {//除每行最右进程
			MPI_Recv(crbuf1, count1, MPI_DOUBLE, iam+1, 20, MPI_COMM_WORLD, &status);
			MPI_Send(csbuf2, count2, MPI_DOUBLE, iam+1, 21, MPI_COMM_WORLD);
		}
		if(iam%npx != 0) {//除每行最左进程
			MPI_Send(csbuf1, count1, MPI_DOUBLE, iam-1, 22, MPI_COMM_WORLD);
			MPI_Recv(crbuf2, count2, MPI_DOUBLE, iam-1, 23, MPI_COMM_WORLD, &status);
		}
	}

	if((iam+1)%npx != 0) {//除了每行最右进程, 解包crbuf1(横向右侧接收)
		count1 = 0;
		for(t = 0; t < 2; t++) {
			for(j = ((iam<npx)? 0 : YSLOPE); j < ((iam<npx*(npy-1)) ? (YSLOPE+iy) : domain_y); j++) {//最下一行，多发送一个下边界，最上层，多发一个上边界
				for(i = XSLOPE+ix; i < XSLOPE+ix+tb+XSLOPE; i++) {
					B[t][j][i] = crbuf1[count1];
					count1++;
				}
			}
		}
	}

	if(iam%npx != 0) {//除了每行最左进程, 解包crbuf2(横向左侧接收)
		count2 = 0;
		for(t = 0; t < 2; t++) {
			for(j = ((iam<npx)? 0 : YSLOPE); j < ((iam<npx*(npy-1)) ? (YSLOPE+iy) : domain_y); j++) {//最下一行，多发送一个下边界，最上层，多发一个上边界
				for(i = 0; i < XSLOPE; i++) {
					B[t][j][i] = crbuf2[count2];
					count2++;
				}
			}
		}
	}


	if(iam >= npx) {//除了最下层进程，打包csbuf1(纵向向下传输)
		count1 = 0;
		for(t = 0; t < 2; t++) {
			for(j = YSLOPE; j < YSLOPE+tb+YSLOPE; j++) {
				for(i = 0; i < domain_x; i++) {
					csbuf3[count1] = B[t][j][i];
					count1++;
				}
			}
		}
	}
	else count1 = (tb+YSLOPE)*domain_x*2;

	if(iam < npx*(npy-1)) {//除最上层进程，打包csbuf2(纵向向上传输)
		count2 = 0;
		for(t = 0; t < 2; t++) { 
			for(j = iy; j < YSLOPE+iy; j++) {
				for(i = 0; i < domain_x; i++) {
					csbuf4[count2] = B[t][j][i];
					count2++;
				}
			}
		}
	}
	else count2 = YSLOPE*domain_x*2;

	if((iam/npx) & 0x1) {//奇数行进程先发送csbuf1，接收csbuf2
		MPI_Send(csbuf3, count1, MPI_DOUBLE, iam-npx, 24, MPI_COMM_WORLD);
		MPI_Recv(crbuf4, count2, MPI_DOUBLE, iam-npx, 25, MPI_COMM_WORLD, &status);
		if(iam < npx*(npy-1)) {//除每列最上进程
			MPI_Recv(crbuf3, count1, MPI_DOUBLE, iam+npx, 26, MPI_COMM_WORLD, &status);
			MPI_Send(csbuf4, count2, MPI_DOUBLE, iam+npx, 27, MPI_COMM_WORLD);
		}
	}
	else {//偶数行进程先接收csbuf1，发送csbuf2
		if(iam < npx*(npy-1)) {//除每列最上进程
			MPI_Recv(crbuf3, count1, MPI_DOUBLE, iam+npx, 24, MPI_COMM_WORLD, &status);
			MPI_Send(csbuf4, count2, MPI_DOUBLE, iam+npx, 25, MPI_COMM_WORLD);
		}
		if(iam >= npx) {//除每列最下进程
			MPI_Send(csbuf3, count1, MPI_DOUBLE, iam-npx, 26, MPI_COMM_WORLD);
			MPI_Recv(crbuf4, count2, MPI_DOUBLE, iam-npx, 27, MPI_COMM_WORLD, &status);
		}
	}

	if(iam < npx*(npy-1)) {//除最上层进程，解包crbuf1(纵向上侧接收)
		count1 = 0;
		for(t = 0; t < 2; t++) {
			for(j = YSLOPE+iy; j < YSLOPE+iy+tb+YSLOPE; j++) {
				for(i = 0; i < domain_x; i++) {
					B[t][j][i] = crbuf3[count1];
					count1++;
				}
			}
		}
	}

	if(iam >= npx) {//除了最下层进程，解包crbuf2(纵向下侧接收)
		count2 = 0;
		for(t = 0; t < 2; t++) {
			for(j = 0; j < YSLOPE; j++) {
				for(i = 0; i < domain_x; i++) {
					B[t][j][i] = crbuf4[count2];
					count2++;
				}
			}
		}
	}
//if(iam==0) printf("before check2, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);

	for (t = 0; t < T; t++) {
		for (y = YSLOPE; y < domain_y-YSLOPE; y++) {
			for (x = XSLOPE; x < domain_x-XSLOPE; x++) {
				kernel(B);
			}
		}
		count1 = 0;
		count2 = 0;
		for(j = YSLOPE; j < domain_y-YSLOPE; j++) {
			for(i = tb+XSLOPE; i < tb+2*XSLOPE; i++){
				csbuf1[count1] = B[(t+1)%2][j][i];
				count1++;
			}
		}
		for(j = YSLOPE; j < domain_y-YSLOPE; j++) {
			for(i = ix; i < ix+XSLOPE; i++){
				csbuf2[count2] = B[(t+1)%2][j][i];
				count2++;
			}
		}
		if((iam%npx) & 0x1) {//奇数列进程先发送csbuf1，接收csbuf2
			MPI_Send(csbuf1, count1, MPI_DOUBLE, iam-1, 30, MPI_COMM_WORLD);
			MPI_Recv(crbuf2, count2, MPI_DOUBLE, iam-1, 31, MPI_COMM_WORLD, &status);
			if((iam+1)%npx != 0) {//除每行最右进程
				MPI_Recv(crbuf1, count1, MPI_DOUBLE, iam+1, 32, MPI_COMM_WORLD, &status);
				MPI_Send(csbuf2, count2, MPI_DOUBLE, iam+1, 33, MPI_COMM_WORLD);
			}
		}
		else {//偶数列进程先接收csbuf1，发送csbuf2
			if((iam+1)%npx != 0) {//除每行最右进程
				MPI_Recv(crbuf1, count1, MPI_DOUBLE, iam+1, 30, MPI_COMM_WORLD, &status);
				MPI_Send(csbuf2, count2, MPI_DOUBLE, iam+1, 31, MPI_COMM_WORLD);
			}
			if(iam%npx != 0) {//除每行最左进程
				MPI_Send(csbuf1, count1, MPI_DOUBLE, iam-1, 32, MPI_COMM_WORLD);
				MPI_Recv(crbuf2, count2, MPI_DOUBLE, iam-1, 33, MPI_COMM_WORLD, &status);
			}
		}
		if((iam+1)%npx != 0) {//除了每行最右进程, 解包crbuf1(横向右侧接收)
			count1 = 0;
			for(j = YSLOPE; j < domain_y-YSLOPE; j++) {
				for(i = domain_x-XSLOPE; i < domain_x; i++){
					B[(t+1)%2][j][i] = crbuf1[count1];
					count1++;
				}
			}
		}
		if(iam%npx != 0) {//除了每行最左进程, 解包crbuf2(横向左侧接收)
			count2 = 0;
			for(j = YSLOPE; j < domain_y-YSLOPE; j++) {
				for(i = 0; i < XSLOPE; i++){
					B[(t+1)%2][j][i] = crbuf2[count2];
					count2++;
				}
			}
		}
//if(iam==0) printf("before check3, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);

		count1 = 0;
		count2 = 0;
		for(j = tb+YSLOPE; j < tb+2*YSLOPE; j++) {
			for(i = 0; i < domain_x; i++){
				csbuf3[count1] = B[(t+1)%2][j][i];
				count1++;
			}
		}
		for(j = iy; j < iy+YSLOPE; j++) {
			for(i = 0; i < domain_x; i++){
				csbuf4[count2] = B[(t+1)%2][j][i];
				count2++;
			}
		}
		if((iam/npx) & 0x1) {//奇数行进程先发送csbuf3，接收csbuf4
			MPI_Send(csbuf3, count1, MPI_DOUBLE, iam-npx, 34, MPI_COMM_WORLD);
			MPI_Recv(crbuf4, count2, MPI_DOUBLE, iam-npx, 35, MPI_COMM_WORLD, &status);
			if(iam < npx*(npy-1)) {//除每列最上进程
				MPI_Recv(crbuf3, count1, MPI_DOUBLE, iam+npx, 36, MPI_COMM_WORLD, &status);
				MPI_Send(csbuf4, count2, MPI_DOUBLE, iam+npx, 37, MPI_COMM_WORLD);
			}
		}
		else {//偶数行进程先接收csbuf3，发送csbuf4
			if(iam < npx*(npy-1)) {//除每列最上进程
				MPI_Recv(crbuf3, count1, MPI_DOUBLE, iam+npx, 34, MPI_COMM_WORLD, &status);
				MPI_Send(csbuf4, count2, MPI_DOUBLE, iam+npx, 35, MPI_COMM_WORLD);
			}
			if(iam >= npx) {//除每列最下进程
				MPI_Send(csbuf3, count1, MPI_DOUBLE, iam-npx, 36, MPI_COMM_WORLD);
				MPI_Recv(crbuf4, count2, MPI_DOUBLE, iam-npx, 37, MPI_COMM_WORLD, &status);
			}
		}
		if(iam < npx*(npy-1)) {//除最上层进程，解包crbuf3(纵向上侧接收)
			count1 = 0;
			for(j = domain_y-YSLOPE; j < domain_y; j++) {
				for(i = 0; i < domain_x; i++){
					B[(t+1)%2][j][i] = crbuf3[count1];
					count1++;
				}
			}
		}
		if(iam >= npx) {//除了最下层进程，解包crbuf4(纵向下侧接收)
			count2 = 0;
			for(j = 0; j < YSLOPE; j++) {
				for(i = 0; i < domain_x; i++){
					B[(t+1)%2][j][i] = crbuf4[count2];
					count2++;
				}
			}
		}
	}
//if(iam==0) printf("before check4, A[0][8][7]=%lf\tA[1][8][7]=%lf\tB[0][8][7]=%lf\tB[1][8][7]=%lf\n",A[0][8][7],A[1][8][7],B[0][8][7],B[1][8][7]);

	for (j = YSLOPE; j < domain_y-YSLOPE; j++) {
		for (i = XSLOPE; i < domain_x-XSLOPE; i++) {
			if(myabs(A[T%2][j][i],B[T%2][j][i]) > TOLERANCE)
				printf("%d:\tNaive[%d][%d] = %f, Check = %f: FAILED!\n", iam, j, i, B[T%2][j][i], A[T%2][j][i]);
		}
	}
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    // free(sbuf1);
    // free(sbuf2);
    // free(rbuf1);
    // free(rbuf2);

  	MPI_Finalize();

}
