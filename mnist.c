/*
*
*IP�̋N�� void X<IP��>_Start(&<�C���X�^���X�ϐ�>)
*IP�̓���I�����f void X<IP��>_IsDone(&<�C���X�^���X�ϐ�>)
*�������|�[�g�ւ̏������� u32 X<IP��>_Write_<�|�[�g��>_Words(&<�C���X�^���X�ϐ�>,<�I�t�Z�b�g>,<�z��>,<��>)
*�������|�[�g�ւ̓ǂݍ��� u32 X<IP��>_Read_<�|�[�g��>_Words(&<�C���X�^���X�ϐ�>,<�I�t�Z�b�g>,<�z��>,<��>)
*/

#include "stdio.h"
#include "xtime_l.h"			/*�N���b�N����ǂݎ�郉�C�u����*/
#include "ff.h"               	/*SDcard�ǂݏ����p���C�u����*/
#include "xlearn_model_v_2.h"   /*BNN���f��Ver2_0*/

XLearn_model_v_2 Instance;		/*�C���X�^���X�ϐ��̐錾*/
XTime startt, endt;

#define Xc 10   /*�o�b�`��*/
#define Xr 98   /*���̓f�[�^�T�C�Y*/
#define W1c 98  /*BFNN�P�w�ڂ̏d�݌W����T�C�Y*/
#define W1r 120 /*BFNN�P�w�ڂ̏d�݌W���s�T�C�Y*/
#define W2c 15  /*BFNN�Q�w�ڂ̏d�݌W����T�C�Y*/
#define W2r 120 /*BFNN�Q�w�ڂ̏d�݌W���s�T�C�Y*/
#define W3c 15  /*BFNN�R�w�ڂ̏d�݌W����T�C�Y*/
#define W3r 120 /*BFNN�R�w�ڂ̏d�݌W���s�T�C�Y*/
#define W4c 15  /*BFNN�S�w�ڂ̏d�݌W����T�C�Y*/
#define W4r 10  /*BFNN�S�w�ڂ̏d�݌W���s�T�C�Y*/
#define c 0.000001 /*0���Z�h�~�W��*/

volatile int b1[W1r]; /*BFNN�P�w�ڂ̃o�C�A�X*/
volatile int b2[W2r]; /*BFNN�Q�w�ڂ̃o�C�A�X*/
volatile int b3[W3r]; /*BFNN�R�w�ڂ̃o�C�A�X*/
volatile int b4[W4r]; /*BFNN�S�w�ڂ̃o�C�A�X*/
volatile unsigned char w1[W1c][W1r]; /*BFNN�P�w�ڂ̏d��*/
volatile unsigned char w2[W2c][W2r]; /*BFNN�Q�w�ڂ̏d��*/
volatile unsigned char w3[W3c][W3r]; /*BFNN�R�w�ڂ̏d��*/
volatile unsigned char w4[W4c][W4r]; /*BFNN�S�w�ڂ̏d��*/

volatile unsigned char set_T[50000][W4r] = { 0 };       /*�������ɐ������ꂽ�������x��*/
#define buff ((volatile unsigned char * ) 0x10000800) 	/*�t�@�C�����璼�ړǂݍ��񂾃f�[�^*/
#define mnist ((volatile unsigned char * ) 0x130003ff)  /*���ړǂݏo���ꂽ�摜�f�[�^*/
#define setdata ((volatile unsigned char * ) 0x16000000)/*�������ɐ������ꂽ�摜�f�[�^*/


void SD_Read(FIL* fil, const TCHAR* filename, UINT size, UINT* nr) {
	FRESULT Res;
	/*�w�肳�ꂽ�t�@�C�����J��*/
	Res = f_open(fil, filename, FA_READ);
	if (Res) {
		printf("ERROR: f_open\n");
		return XST_FAILURE;
	}
	/*�J�����t�@�C���̃f�[�^��ǂݍ���*/
	Res = f_read(fil, buff, size, nr);
	if (Res) {
		printf("ERROR: f_read\n");
		return XST_FAILURE;
	}
	/*�t�@�C�������*/
	Res = f_close(fil);
	if (Res != FR_OK) {
		printf("ERROR: f_close\n");
		return XST_FAILURE;
	}

}

double my_pow(double x, int n)
{/*
 x��n����v�Z
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
 logx���v�Z
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
	/*�e�p�����[�^�̏�����*/
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
	/*BNN���f���̊w�K�E���_���s*/
	/*�w�K/���_�̎w��*/
	XLearn_model_v_2_Set_k(&Instance, k);
	/*�o�b�`�P�ʂł̐������x���̏�������*/
	XLearn_model_v_2_Write_T_Bytes(&Instance, 0, T, W4r*Xc);
	/*�o�b�`�P�ʂł̌P��/���؃f�[�^�̏�������*/
	XLearn_model_v_2_Write_in_r_Bytes(&Instance, 0, data, Xr*Xc);
	/*�w�K/���_�J�n*/
	XLearn_model_v_2_Start(&Instance);
	/*����I���܂őҋ@*/
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
	/*�����l�p�����[�^�̏�������*/
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
	�����G���g���s�[�덷�֐�����loss�̌v�Z
	*/
	double res[Xc][W4r];
	double k = 0;
	u8 right[Xc][W4r];
	/*�\���l�̓ǂݎ��*/
	XLearn_model_v_2_Read_out_r_Words(&Instance, 0, res, W4r * 2 * Xc);
	/*�������x���̓ǂݎ��*/
	XLearn_model_v_2_Read_T_Bytes(&Instance, 0, right, W4r*Xc);
	for (int i = 0; i<Xc; i++) {
		for (int j = 0; j<W4r; j++) {
			k -= right[i][j] ? my_log(res[i][j] + c) : 0;
		}
	}
	return k / Xc;/*�o�b�`���Ŋ���A���ς����߂�*/
}

double acc() {
	/*
	���𗦂̌v�Z
	*/
	double res[Xc][W4r];
	u8 right[Xc][W4r];
	double maxp = 0;
	u8 lp, lr;
	double ac = 0;
	/*�\���l�̓ǂݎ��*/
	XLearn_model_v_2_Read_out_r_Words(&Instance, 0, res, W4r * 2 * Xc);
	/*�������x���̓ǂݎ��*/
	XLearn_model_v_2_Read_T_Bytes(&Instance, 0, right, W4r*Xc);
	for (int i = 0; i<Xc; i++)
	{
		maxp = 0;
		for (int u = 0; u<W4r; u++)
		{   /*��ԑ傫�ȗ\���l����\�z�����N���X�����߂�*/
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
	ac /= Xc;/*�o�b�`���Ŋ���A���ς����߂�*/
	return ac;
}

int main() {
	/*MNIST�̎菑�����������摜�̃t�@�C����*/
	char dataFile[32] = "train-images.idx3-ubyte";
	/*MNIST�̎菑�����������̐������x���̃t�@�C����*/
	char labelFile[32] = "train-labels.idx1-ubyte";
	FIL fil;
	FATFS fatfs;
	FRESULT Res;
	TCHAR *Path = "0:/";
	UINT NumBytesRead;
	/*�摜�t�@�C���f�[�^�̃T�C�Y*/
	u32 dataSize = 784 * 60000 + 16;
	/*�������x���f�[�^�̃T�C�Y*/
	u32 labelSize = 10 * 60000 + 8;

	printf("Now Loading...\n");

	/******mount_SDcard*****/
	Res = f_mount(&fatfs, Path, 0);
	if (Res != FR_OK) {
		printf("ERROR: f_mount\n");
		return XST_FAILURE;
	}

	/*MNIST�̎菑�����������摜�̓ǂݍ���*/
	SD_Read(&fil, dataFile, dataSize, &NumBytesRead);

	for (int i = 2; i<NumBytesRead / 8; i++) {
		/*2byte�ȍ~�̃f�[�^���Q�l�����đ��*/
		mnist[i - 2] = (buff[i * 8]>128 ? 0x80 : 0x00) |
			(buff[(i * 8) + 1]>128 ? 0x40 : 0x00) |
			(buff[(i * 8) + 2]>128 ? 0x20 : 0x00) |
			(buff[(i * 8) + 3]>128 ? 0x10 : 0x00) |
			(buff[(i * 8) + 4]>128 ? 0x08 : 0x00) |
			(buff[(i * 8) + 5]>128 ? 0x04 : 0x00) |
			(buff[(i * 8) + 6]>128 ? 0x02 : 0x00) |
			(buff[(i * 8) + 7]>128 ? 0x01 : 0x00);
	}

	/*�������x���̓ǂݍ���*/
	SD_Read(&fil, labelFile, labelSize, &NumBytesRead);

	uint label_count[10] = { 0 };
	/*1byte�ȍ~�̃f�[�^���Q�l�����đ��*/
	for (int i = 8; i<60008; i++) {
		++label_count[buff[i]];
		if (label_count[buff[i]]<5000) {
			for (int n = 0; n<Xr; n++)
				/*�����ʂɃf�[�^�𕪂���*/
				setdata[Xr * (5000 * buff[i] + label_count[buff[i]]) + n] = mnist[Xr * (i - 8) + n];
			set_T[5000 * buff[i] + label_count[buff[i]]][buff[i]] = 1;
		}
	}
	/*�p�����[�^������*/
	init(b1, b2, b3, b4, w1, w2, w3, w4);

	if (XLearn_model_v_2_Initialize(&Instance, XPAR_XLEARN_MODEL_V_2_0_DEVICE_ID) != XST_SUCCESS)/*�C���X�^���X�̏�����*/ {
		printf("init_error!\n");
		return XST_FAILURE;
	}
	/*�����������p�����[�^����H�ɏ�������*/
	learn_init((u32)b1, (u32)b2, (u32)b3, (u32)b4, (u32)w1, (u32)w2, (u32)w3, (u32)w4, &(set_T[0]), &(mnist[Xr * 0]));

	u8 rm[Xr*Xc];		/*�o�b�`�P�ʂ̌P���p�摜�f�[�^*/
	u8 rt[10][Xc];	/*�o�b�`�P�ʂ̌P���p�������x��*/
	u8 k_rm[Xr+Xc];	/*�o�b�`�P�ʂ̌��ؗp�摜�f�[�^*/
	u8 k_rt[10][Xc];/*�o�b�`�P�ʂ̌��ؗp�������x��*/

#define train_range 6000	/*�P���f�[�^������6000��*/
#define test_range 1000		/*���؃f�[�^������1000��*/
#define epochs 100			/*100�G�|�b�N�w�K���s��*/

	for (int z = 1; z<epochs; z++) {
		double mid = 0, tmid = 0, tac = 0, lac = 0;
		XTime_GetTime(&startt);/*�J�n���̑��N���b�N�����L�^*/
							   /*�P���f�[�^�̊w�K*/
		for (int f = 0; f<train_range / Xc; f++) {
			for (int m = 0; m<Xc; m++) {
				/*0�`600�̗����𐶐�*/
				u32 ra = rand() % train_range / Xc;
				/*0�`10�̗����𐶐�*/
				u8 ra_range = rand() % Xc;
				for (int n = 0; n<Xr; n++)
					/*�o�b�`�P�ʂœ��͂���P���f�[�^�������_���ɐ���*/
					rm[(Xr * ra_range) + n] = setdata[(Xr * (5000 * m + ra)) + n];
				for (int n = 0; n<Xc; n++)
					/*�o�b�`�P�ʂœ��͂��鐳�����x���������_���ɐ���*/
					rt[ra_range][n] = set_T[5000 * m + ra][n];
			}
			/*�w�K���s*/
			learn(0, rt, rm);
			lac += acc(); /*���𗦂����Z*/
			mid += loss();/*Loss�����Z*/
			}
		XTime_GetTime(&endt);
					printf("learn : %8.3f[pcs] ", 6000.0/((endt - startt) / 325000000.0));

		/*���؃f�[�^�̐��_*/
		XTime_GetTime(&startt);
		for (int q = 0; q<test_range / Xc; q++) {
			for (int m = 0; m<Xc; m++) {
				for (int n = 0; n<Xr; n++)
					/*�o�b�`�P�ʂœ��͂��錟�؃f�[�^�𐶐�*/
					k_rm[(Xr * m) + n] = setdata[(Xr * (5000 * m + q + train_range / Xc)) + n];
				for (int n = 0; n<Xc; n++)
					/*�o�b�`�P�ʂœ��͂��鐳�����x���𐶐�*/
					k_rt[m][n] = set_T[5000 * m + q + train_range / Xc][n];
			}

			/*���_���s*/
			learn(1, k_rt, k_rm);
			tac += acc();  /*���𗦂����Z*/
			tmid += loss();/*Loss�����Z*/
		}
		XTime_GetTime(&endt);
					printf("prediction : %8.3f[pcs] ", 1000.0/((endt - startt) / 325000000.0));

		/*���ϒl���Z�o*/
		tac /= test_range / Xc;
		lac /= train_range / Xc;
		tmid /= test_range / Xc;
		mid /= train_range / Xc;

		XTime_GetTime(&endt);				/*1epoch�I�����̑��N���b�N�����L�^*/
		/*���v���Ԃ�\��*/
		printf("processing time : %8.3f[s]_", (double)(endt - startt) / 325000000);
		printf("%depoch", z);				/*�G�|�b�N����\��*/
		printf("  loss:%8.3f", mid);		/*�G�|�b�N���ɕ��ς��Ƃ����w�K�f�[�^��loss�l��\��*/
		printf("  acc:%8.3f", lac);			/*�G�|�b�N���ɕ��ς��Ƃ����w�K�f�[�^�̐��𗦂�\��*/
		printf("  test_loss:%8.3f", tmid);	/*�G�|�b�N���ɕ��ς��Ƃ������؃f�[�^��loss�l��\��*/
		printf("  test_acc:%8.3f\n", tac);	/*�G�|�b�N���ɕ��ς��Ƃ������؃f�[�^�̐��𗦂�\��*/
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
