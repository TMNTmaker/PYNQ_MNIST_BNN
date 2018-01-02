/*
*
*IPの起動 void X<IP名>_Start(&<インスタンス変数>)
*IPの動作終了判断 void X<IP名>_IsDone(&<インスタンス変数>)
*メモリポートへの書き込み u32 X<IP名>_Write_<ポート名>_Words(&<インスタンス変数>,<オフセット>,<配列>,<個数>)
*メモリポートへの読み込み u32 X<IP名>_Read_<ポート名>_Words(&<インスタンス変数>,<オフセット>,<配列>,<個数>)
*/

#include "stdio.h"
#include "xtime_l.h"			/*クロック数を読み取るライブラリ*/
#include "ff.h"               	/*SDcard読み書き用ライブラリ*/
#include "xlearn_model_v_2.h"   /*BNNモデルVer2_0*/

XLearn_model_v_2 Instance;		/*インスタンス変数の宣言*/
XTime startt, endt;

#define Xc 10   /*バッチ数*/
#define Xr 98   /*入力データサイズ*/
#define W1c 98  /*BFNN１層目の重み係数列サイズ*/
#define W1r 120 /*BFNN１層目の重み係数行サイズ*/
#define W2c 15  /*BFNN２層目の重み係数列サイズ*/
#define W2r 120 /*BFNN２層目の重み係数行サイズ*/
#define W3c 15  /*BFNN３層目の重み係数列サイズ*/
#define W3r 120 /*BFNN３層目の重み係数行サイズ*/
#define W4c 15  /*BFNN４層目の重み係数列サイズ*/
#define W4r 10  /*BFNN４層目の重み係数行サイズ*/
#define c 0.000001 /*0除算防止係数*/

volatile int b1[W1r]; /*BFNN１層目のバイアス*/
volatile int b2[W2r]; /*BFNN２層目のバイアス*/
volatile int b3[W3r]; /*BFNN３層目のバイアス*/
volatile int b4[W4r]; /*BFNN４層目のバイアス*/
volatile unsigned char w1[W1c][W1r]; /*BFNN１層目の重み*/
volatile unsigned char w2[W2c][W2r]; /*BFNN２層目の重み*/
volatile unsigned char w3[W3c][W3r]; /*BFNN３層目の重み*/
volatile unsigned char w4[W4c][W4r]; /*BFNN４層目の重み*/

volatile unsigned char set_T[50000][W4r] = { 0 };       /*数字毎に整理された正解ラベル*/
#define buff ((volatile unsigned char * ) 0x10000800) 	/*ファイルから直接読み込んだデータ*/
#define mnist ((volatile unsigned char * ) 0x130003ff)  /*直接読み出された画像データ*/
#define setdata ((volatile unsigned char * ) 0x16000000)/*数字毎に整理された画像データ*/


void SD_Read(FIL* fil, const TCHAR* filename, UINT size, UINT* nr) {
	FRESULT Res;
	/*指定されたファイルを開く*/
	Res = f_open(fil, filename, FA_READ);
	if (Res) {
		printf("ERROR: f_open\n");
		return XST_FAILURE;
	}
	/*開いたファイルのデータを読み込む*/
	Res = f_read(fil, buff, size, nr);
	if (Res) {
		printf("ERROR: f_read\n");
		return XST_FAILURE;
	}
	/*ファイルを閉じる*/
	Res = f_close(fil);
	if (Res != FR_OK) {
		printf("ERROR: f_close\n");
		return XST_FAILURE;
	}

}

double my_pow(double x, int n)
{/*
 xのn乗を計算
 */
	int i;
	double pow_result = 1;

	if (n == 0)
		return 1;
	else
	{
		for (i = 0; i < n; i++)
		{
			pow_result *= x;
		}
		return pow_result;
	}
}

double my_log(double x)
{/*
 logxを計算
 */
	int i;
	double result, result1, result2;

	x -= 1;
	result1 = 0;
	result2 = 0;

	for (i = 1; i <= 40; i++)
	{
		if (i % 2 == 1)
			result1 += my_pow(x, i) / i;
		else

			result2 += my_pow(x, i) / i;
	}

	return result1 - result2;
}

void init(volatile int b1i[W1r],
	volatile int b2i[W2r],
	volatile int b3i[W3r],
	volatile int b4i[W4r],
	volatile unsigned char w1i[W1c][W1r],
	volatile unsigned char w2i[W2c][W2r],
	volatile unsigned char w3i[W3c][W3r],
	volatile unsigned char w4i[W4c][W4r])
{
	/*各パラメータの初期化*/
	int i, u;
	for (i = 0; i<W1r; i++) b1i[i] = ((rand() & 0xff)>128 ? -1 : 1)*(rand() & 0xff);
	for (i = 0; i<W2r; i++) b2i[i] = ((rand() & 0xff)>128 ? -1 : 1)*(rand() & 0xff);
	for (i = 0; i<W3r; i++) b3i[i] = ((rand() & 0xff)>128 ? -1 : 1)*(rand() & 0xff);
	for (i = 0; i<W4r; i++) b4i[i] = ((rand() & 0xff)>128 ? -1 : 1)*(rand() & 0xff);

	for (i = 0; i<W1c; i++)
		for (u = 0; u<W1r; u++)
			w1i[i][u] = (rand() & 0xff);
	for (i = 0; i<W2c; i++)
		for (u = 0; u<W2r; u++)
			w2i[i][u] = (rand() & 0xff);
	for (i = 0; i<W3c; i++) {
		for (u = 0; u<W3r; u++) {
			w3i[i][u] = (rand() & 0xff);
		}
	}
	for (i = 0; i<W4c; i++) {
		for (u = 0; u<W4r; u++) {
			w4i[i][u] = (rand() & 0xff);
		}
	}
	return;
}

void learn(u32 k, u32 T, u32 data) {
	/*BNNモデルの学習・推論実行*/
	/*学習/推論の指定*/
	XLearn_model_v_2_Set_k(&Instance, k);
	/*バッチ単位での正解ラベルの書き込み*/
	XLearn_model_v_2_Write_T_Bytes(&Instance, 0, T, W4r*Xc);
	/*バッチ単位での訓練/検証データの書き込み*/
	XLearn_model_v_2_Write_in_r_Bytes(&Instance, 0, data, Xr*Xc);
	/*学習/推論開始*/
	XLearn_model_v_2_Start(&Instance);
	/*動作終了まで待機*/
	while (XLearn_model_v_2_IsDone(&Instance) == 0);
}

void learn_init(u32 b1i,
	u32 b2i,
	u32 b3i,
	u32 b4i,
	u32 w1i,
	u32 w2i,
	u32 w3i,
	u32 w4i,
	u32 T,
	u32 data
) {
	/*初期値パラメータの書き込み*/
	XLearn_model_v_2_Write_bias1_Bytes(&Instance, 0, b1i, W1r * 4);
	XLearn_model_v_2_Write_bias2_Bytes(&Instance, 0, b2i, W2r * 4);
	XLearn_model_v_2_Write_bias3_Bytes(&Instance, 0, b3i, W3r * 4);
	XLearn_model_v_2_Write_bias4_Bytes(&Instance, 0, b4i, W4r * 4);
	XLearn_model_v_2_Write_weight1_Bytes(&Instance, 0, w1i, W1c*W1r);
	XLearn_model_v_2_Write_weight2_Bytes(&Instance, 0, w2i, W2c*W2r);
	XLearn_model_v_2_Write_weight3_Bytes(&Instance, 0, w3i, W3c*W3r);
	XLearn_model_v_2_Write_weight4_Bytes(&Instance, 0, w4i, W4c*W4r);
	XLearn_model_v_2_Write_T_Bytes(&Instance, 0, T, W4r*Xc);
	XLearn_model_v_2_Write_in_r_Bytes(&Instance, 0, data, Xr*Xc);
}

double loss() {
	/*
	交差エントロピー誤差関数からlossの計算
	*/
	double res[Xc][W4r];
	double k = 0;
	u8 right[Xc][W4r];
	/*予測値の読み取り*/
	XLearn_model_v_2_Read_out_r_Words(&Instance, 0, res, W4r * 2 * Xc);
	/*正解ラベルの読み取り*/
	XLearn_model_v_2_Read_T_Bytes(&Instance, 0, right, W4r*Xc);
	for (int i = 0; i<Xc; i++) {
		for (int j = 0; j<W4r; j++) {
			k -= right[i][j] ? my_log(res[i][j] + c) : 0;
		}
	}
	return k / Xc;/*バッチ数で割り、平均を求める*/
}

double acc() {
	/*
	正解率の計算
	*/
	double res[Xc][W4r];
	u8 right[Xc][W4r];
	double maxp = 0;
	u8 lp, lr;
	double ac = 0;
	/*予測値の読み取り*/
	XLearn_model_v_2_Read_out_r_Words(&Instance, 0, res, W4r * 2 * Xc);
	/*正解ラベルの読み取り*/
	XLearn_model_v_2_Read_T_Bytes(&Instance, 0, right, W4r*Xc);
	for (int i = 0; i<Xc; i++)
	{
		maxp = 0;
		for (int u = 0; u<W4r; u++)
		{   /*一番大きな予測値から予想したクラスを求める*/
			if (maxp<res[i][u])
			{
				maxp = res[i][u];
				lp = u;
			}
			if (right[i][u] == 1)
				lr = u;
		}

		if (lp == lr) { ac++; }
	}
	ac /= Xc;/*バッチ数で割り、平均を求める*/
	return ac;
}

int main() {
	/*MNISTの手書き数字文字画像のファイル名*/
	char dataFile[32] = "train-images.idx3-ubyte";
	/*MNISTの手書き数字文字の正解ラベルのファイル名*/
	char labelFile[32] = "train-labels.idx1-ubyte";
	FIL fil;
	FATFS fatfs;
	FRESULT Res;
	TCHAR *Path = "0:/";
	UINT NumBytesRead;
	/*画像ファイルデータのサイズ*/
	u32 dataSize = 784 * 60000 + 16;
	/*正解ラベルデータのサイズ*/
	u32 labelSize = 10 * 60000 + 8;

	printf("Now Loading...\n");

	/******mount_SDcard*****/
	Res = f_mount(&fatfs, Path, 0);
	if (Res != FR_OK) {
		printf("ERROR: f_mount\n");
		return XST_FAILURE;
	}

	/*MNISTの手書き数字文字画像の読み込み*/
	SD_Read(&fil, dataFile, dataSize, &NumBytesRead);

	for (int i = 2; i<NumBytesRead / 8; i++) {
		/*2byte以降のデータを２値化して代入*/
		mnist[i - 2] = (buff[i * 8]>128 ? 0x80 : 0x00) |
			(buff[(i * 8) + 1]>128 ? 0x40 : 0x00) |
			(buff[(i * 8) + 2]>128 ? 0x20 : 0x00) |
			(buff[(i * 8) + 3]>128 ? 0x10 : 0x00) |
			(buff[(i * 8) + 4]>128 ? 0x08 : 0x00) |
			(buff[(i * 8) + 5]>128 ? 0x04 : 0x00) |
			(buff[(i * 8) + 6]>128 ? 0x02 : 0x00) |
			(buff[(i * 8) + 7]>128 ? 0x01 : 0x00);
	}

	/*正解ラベルの読み込み*/
	SD_Read(&fil, labelFile, labelSize, &NumBytesRead);

	uint label_count[10] = { 0 };
	/*1byte以降のデータを２値化して代入*/
	for (int i = 8; i<60008; i++) {
		++label_count[buff[i]];
		if (label_count[buff[i]]<5000) {
			for (int n = 0; n<Xr; n++)
				/*数字別にデータを分ける*/
				setdata[Xr * (5000 * buff[i] + label_count[buff[i]]) + n] = mnist[Xr * (i - 8) + n];
			set_T[5000 * buff[i] + label_count[buff[i]]][buff[i]] = 1;
		}
	}
	/*パラメータ初期化*/
	init(b1, b2, b3, b4, w1, w2, w3, w4);

	if (XLearn_model_v_2_Initialize(&Instance, XPAR_XLEARN_MODEL_V_2_0_DEVICE_ID) != XST_SUCCESS)/*インスタンスの初期化*/ {
		printf("init_error!\n");
		return XST_FAILURE;
	}
	/*初期化したパラメータを回路に書き込む*/
	learn_init((u32)b1, (u32)b2, (u32)b3, (u32)b4, (u32)w1, (u32)w2, (u32)w3, (u32)w4, &(set_T[0]), &(mnist[Xr * 0]));

	u8 rm[Xr*Xc];		/*バッチ単位の訓練用画像データ*/
	u8 rt[10][Xc];	/*バッチ単位の訓練用正解ラベル*/
	u8 k_rm[Xr+Xc];	/*バッチ単位の検証用画像データ*/
	u8 k_rt[10][Xc];/*バッチ単位の検証用正解ラベル*/

#define train_range 6000	/*訓練データ枚数は6000枚*/
#define test_range 1000		/*検証データ枚数は1000枚*/
#define epochs 100			/*100エポック学習を行う*/

	for (int z = 1; z<epochs; z++) {
		double mid = 0, tmid = 0, tac = 0, lac = 0;
		XTime_GetTime(&startt);/*開始時の総クロック数を記録*/
							   /*訓練データの学習*/
		for (int f = 0; f<train_range / Xc; f++) {
			for (int m = 0; m<Xc; m++) {
				/*0～600の乱数を生成*/
				u32 ra = rand() % train_range / Xc;
				/*0～10の乱数を生成*/
				u8 ra_range = rand() % Xc;
				for (int n = 0; n<Xr; n++)
					/*バッチ単位で入力する訓練データをランダムに生成*/
					rm[(Xr * ra_range) + n] = setdata[(Xr * (5000 * m + ra)) + n];
				for (int n = 0; n<Xc; n++)
					/*バッチ単位で入力する正解ラベルをランダムに生成*/
					rt[ra_range][n] = set_T[5000 * m + ra][n];
			}
			/*学習実行*/
			learn(0, rt, rm);
			lac += acc(); /*正解率を加算*/
			mid += loss();/*Lossを加算*/
			}
		XTime_GetTime(&endt);
					printf("learn : %8.3f[pcs] ", 6000.0/((endt - startt) / 325000000.0));

		/*検証データの推論*/
		XTime_GetTime(&startt);
		for (int q = 0; q<test_range / Xc; q++) {
			for (int m = 0; m<Xc; m++) {
				for (int n = 0; n<Xr; n++)
					/*バッチ単位で入力する検証データを生成*/
					k_rm[(Xr * m) + n] = setdata[(Xr * (5000 * m + q + train_range / Xc)) + n];
				for (int n = 0; n<Xc; n++)
					/*バッチ単位で入力する正解ラベルを生成*/
					k_rt[m][n] = set_T[5000 * m + q + train_range / Xc][n];
			}

			/*推論実行*/
			learn(1, k_rt, k_rm);
			tac += acc();  /*正解率を加算*/
			tmid += loss();/*Lossを加算*/
		}
		XTime_GetTime(&endt);
					printf("prediction : %8.3f[pcs] ", 1000.0/((endt - startt) / 325000000.0));

		/*平均値を算出*/
		tac /= test_range / Xc;
		lac /= train_range / Xc;
		tmid /= test_range / Xc;
		mid /= train_range / Xc;

		XTime_GetTime(&endt);				/*1epoch終了時の総クロック数を記録*/
		/*所要時間を表示*/
		printf("processing time : %8.3f[s]_", (double)(endt - startt) / 325000000);
		printf("%depoch", z);				/*エポック数を表示*/
		printf("  loss:%8.3f", mid);		/*エポック毎に平均をとった学習データのloss値を表示*/
		printf("  acc:%8.3f", lac);			/*エポック毎に平均をとった学習データの正解率を表示*/
		printf("  test_loss:%8.3f", tmid);	/*エポック毎に平均をとった検証データのloss値を表示*/
		printf("  test_acc:%8.3f\n", tac);	/*エポック毎に平均をとった検証データの正解率を表示*/
	}
		XLearn_model_v_2_Read_bias1_Bytes(&Instance, 0, b1, W1r * 4);
		XLearn_model_v_2_Read_bias2_Bytes(&Instance, 0, b2, W2r * 4);
		XLearn_model_v_2_Read_bias3_Bytes(&Instance, 0, b3, W3r * 4);
		XLearn_model_v_2_Read_bias4_Bytes(&Instance, 0, b4, W4r * 4);
        for(int i=0; i<W1r;i++){printf("b1:%d\n",b1[i]);}
        for(int i=0; i<W2r;i++){printf("b2:%d\n",b2[i]);}
        for(int i=0; i<W3r;i++){printf("b3:%d\n",b3[i]);}
        for(int i=0; i<W4r;i++){printf("b4:%d\n",b4[i]);}

	printf("finish\n");
	return 0;
}
