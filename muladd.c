#include <ap_cint.h> /*�C�Ӑ��x�����f�[�^�^�C�v�̃��C�u����*/
#include <math.h>
#include "stdio.h"

#define bs 0.9     /*Batch Deviation�̔����l*/
#define bl4 16     /*BFNN4�w�ڂ̊w�K��(1/bl4)*/
#define bl3 256    /*BFNN�R�w�ڂ̊w�K��(1/bl3)*/
#define bl2 4096   /*BFNN�Q�w�ڂ̊w�K��(1/bl2)*/
#define bl1 16384  /*BFNN�P�w�ڂ̊w�K��(1/bl1)*/
#define Xc 10      /*�o�b�`��*/
#define Xr 98      /*���̓f�[�^�T�C�Y*/
#define W1c 98     /*BFNN�P�w�ڂ̏d�݌W����T�C�Y*/
#define W1r 120    /*BFNN�P�w�ڂ̏d�݌W���s�T�C�Y*/
#define W2c 15     /*BFNN�Q�w�ڂ̏d�݌W����T�C�Y*/
#define W2r 120    /*BFNN�Q�w�ڂ̏d�݌W���s�T�C�Y*/
#define W3c 15     /*BFNN�R�w�ڂ̏d�݌W����T�C�Y*/
#define W3r 120    /*BFNN�R�w�ڂ̏d�݌W���s�T�C�Y*/
#define W4c 15     /*BFNN�S�w�ڂ̏d�݌W����T�C�Y*/
#define W4r 10     /*BFNN�S�w�ڂ̏d�݌W���s�T�C�Y*/
#define c 0.000001 /*0���Z�h�~�W��*/


uint4 popcnt_8(uint8 bits)
{/*�P�̃r�b�g���𐔂���*/
	bits = (bits & 0x55) + (bits >> 1 & 0x55);
	bits = (bits & 0x33) + (bits >> 2 & 0x33);
	return bits = (bits & 0x0f) + (bits >> 4 & 0x0f);
}

/*
*b_a:STE�ƒ�Ď�@OL�̏��`�Ԃ��s���֐�
*�Ϙa���ꂽ���͒l>0�̂Ƃ�1,
*�ۂȂ��-1(�����ł�0�ɒu������)
*/
void b_a1(int17 in[Xc][W1r], uint8 out[Xc][W1r / 8]) {
	uint7 j, i;
L1:for (i = 0; i<Xc; i++) {
L2:for (j = 0; j<(W1r / 8); j++) {
	out[i][j] = (in[i][(j * 8)]>0 ? 0x80 : 0x00) | (in[i][(j * 8) + 1]>0 ? 0x40 : 0x00) | (in[i][(j * 8) + 2]>0 ? 0x20 : 0x00)
		| (in[i][(j * 8) + 3]>0 ? 0x10 : 0x00) | (in[i][(j * 8) + 4]>0 ? 0x08 : 0x00) | (in[i][(j * 8) + 5]>0 ? 0x04 : 0x00)
		| (in[i][(j * 8) + 6]>0 ? 0x02 : 0x00) | (in[i][(j * 8) + 7]>0 ? 0x01 : 0x00);
}
}
}

void b_a2(int17 in[Xc][W2r], uint8 out[Xc][W2r / 8]) {
	uint5 j, i;
L1:for (i = 0; i<Xc; i++) {
L2:for (j = 0; j<(W2r / 8); j++) {
	out[i][j] = (in[i][(j * 8)]>0 ? 0x80 : 0x00) | (in[i][(j * 8) + 1]>0 ? 0x40 : 0x00) | (in[i][(j * 8) + 2]>0 ? 0x20 : 0x00)
		| (in[i][(j * 8) + 3]>0 ? 0x10 : 0x00) | (in[i][(j * 8) + 4]>0 ? 0x08 : 0x00) | (in[i][(j * 8) + 5]>0 ? 0x04 : 0x00)
		| (in[i][(j * 8) + 6]>0 ? 0x02 : 0x00) | (in[i][(j * 8) + 7]>0 ? 0x01 : 0x00);
}
}
}

void b_a3(int17 in[Xc][W3r], uint8 out[Xc][W3r / 8]) {
	uint5 j, i;
L1:for (i = 0; i<Xc; i++) {
L2:for (j = 0; j<(W3r / 8); j++) {
	out[i][j] = (in[i][(j * 8)]>0 ? 0x80 : 0x00) | (in[i][(j * 8) + 1]>0 ? 0x40 : 0x00) | (in[i][(j * 8) + 2]>0 ? 0x20 : 0x00)
		| (in[i][(j * 8) + 3]>0 ? 0x10 : 0x00) | (in[i][(j * 8) + 4]>0 ? 0x08 : 0x00) | (in[i][(j * 8) + 5]>0 ? 0x04 : 0x00)
		| (in[i][(j * 8) + 6]>0 ? 0x02 : 0x00) | (in[i][(j * 8) + 7]>0 ? 0x01 : 0x00);
}
}
}

/*
*b_a_bw:STE�̋t�`�d���s���֐�
*���͒l���J���(-1,1)�̂Ƃ��O�w����̌��z�ɂP�|����
*�ۂȂ��0
*/
void b_a1_bw(int17 x[Xc][W1r], int32 dy[Xc][W2c * 8]) {
	uint9 j,i;
L1:for (i = 0; i<Xc; i++){
L2:for (j = 0; j<W2c * 8; j++) {
	dy[i][j] = (x[i][j]>-1) && (x[i][j]<1) ? dy[i][j] : 0;
}
}
}

void b_a2_bw(int17 x[Xc][W2r], int32 dy[Xc][W3c * 8]) {
	uint7 j,i;
L1:for (i = 0; i<Xc; i++){
L2:for (j = 0; j<W3c * 8; j++) {
	dy[i][j] = (x[i][j]>-1) && (x[i][j]<1) ? dy[i][j] : 0;
}
}
}

void b_a3_bw(int17 x[Xc][W3r], int32 dy[Xc][W4c * 8]) {
	uint7 j,i;
L1:for (i = 0; i<Xc; i++){
L2:for (j = 0; j<W4c * 8; j++) {
	dy[i][j] = (x[i][j]>-1) && (x[i][j]<1) ? dy[i][j] : 0;
}
}
}

/*
*leaky_relu1_bw:OL�̋t�`�d���s���֐�
*1>���͒l>-1�̂Ƃ��O�w����̌��z�ɂP�|����
*�ۂȂ�ΑO�w����̌��z��0.5�|����
*/
void leaky_relu1_bw(int17 x[Xc][W1r], int32 dy[Xc][W2c * 8]) {
	uint9 j,i;
L1:for (i = 0; i<Xc; i++){
L2:for (j = 0; j<W2c * 8; j++) {
	dy[i][j] = (x[i][j]>=-1) ? dy[i][j] : 0;
}
}
}

void leaky_relu2_bw(int17 x[Xc][W2r], int32 dy[Xc][W3c * 8]) {
	uint7 j,i;
L1:for (i = 0; i<Xc; i++){
L2:for (j = 0; j<W3c * 8; j++) {
	dy[i][j] = (x[i][j]>=-1) ? dy[i][j] : 0;
}
}
}

void leaky_relu3_bw(int17 x[Xc][W3r], int32 dy[Xc][W4c * 8]) {
	uint7 j,i;
L1:for (i = 0; i<Xc; i++){
L2:for (j = 0; j<W4c * 8; j++) {
	dy[i][j] = (x[i][j]>=-1) ? dy[i][j] : 0;
}
}
}


/*bd:�o�b�`���ɓ��͒l�̕΍��𓱏o����֐�*/
void bd1(int17 in[Xc][W1r], int17 out[Xc][W1r]) {
	uint10 i, u;
	uint17 ub[W1r];
ub1:for (u = 0; u<W1r; u++) {
	int32 sum = 0;
ub2:for (i = 0; i<Xc; i++) {
	sum += in[i][u];
}
	ub[u] = sum / Xc;			/*���ϒl�Z�o*/
}
ab1:for (u = 0; u<W1r; u++) {
	int32 sum = 0;
ab2:for (i = 0; i<Xc; i++) {
	out[i][u] = (in[i][u] - ub[u]);/*�΍��Z�o*/
}
}

}
void bd2(int17 in[Xc][W2r], int17 out[Xc][W2r]) {
	uint10 i, u;
	uint17 ub[W2r];
ub1:for (u = 0; u<W2r; u++) {
	int32 sum = 0;
ub2:for (i = 0; i<Xc; i++) {
	sum += in[i][u];
}
	ub[u] = sum / Xc;
}
ab1:for (u = 0; u<W2r; u++) {
	int32 sum = 0;
ab2:for (i = 0; i<Xc; i++) {
	out[i][u] = (in[i][u] - ub[u]);
}
}

}
void bd3(int17 in[Xc][W3r], int17 out[Xc][W3r]) {
	uint10 i, u;
	uint17 ub[W3r];
ub1:for (u = 0; u<W3r; u++) {
	int32 sum = 0;
ub2:for (i = 0; i<Xc; i++) {
	sum += in[i][u];
}
	ub[u] = sum / Xc;
}
ab1:for (u = 0; u<W3r; u++) {
	int32 sum = 0;
ab2:for (i = 0; i<Xc; i++) {
	out[i][u] = (in[i][u] - ub[u]);
}
}

}

/*
*bd_bw:BD�̋t�`�d���s���֐�
*�����W���̓o�b�`���݂̂Ɉˑ����邽�ߒ萔bs���|����
*/
void bd1_bw(int32 dy[Xc][W1r]) {
	uint10 i, u;
	for (i = 0; i<Xc; i++) {
		for (u = 0; u<W1r; u++) {
			dy[i][u] = dy[i][u] * bs;
		}
	}

}
void bd2_bw(int32 dy[Xc][W2r]) {
	uint10 i, u;
	for (i = 0; i<Xc; i++) {
		for (u = 0; u<W2r; u++) {
			dy[i][u] = dy[i][u] * bs;

		}
	}

}
void bd3_bw(int32 dy[Xc][W3r]) {
	uint10 i, u;
	for (i = 0; i<Xc; i++) {
		for (u = 0; u<W3r; u++) {
			dy[i][u] = dy[i][u] * bs;

		}
	}

}


void sf(int17 in[Xc][W4r], double out[Xc][W4r]) {
	/*�\�t�g�}�b�N�X�֐�*/
	uint4 i, j;
sc1:for (j = 0; j<Xc; j++) {
	int17 max = 0; double s = 0;
ml1:for (i = 0; i<W4r; i++) {
	in[j][i]>max ? max = in[j][i] : max;
}

sl1:for (i = 0; i<W4r; i++) {
	s += exp(in[j][i] - max);/*�I�[�o�[�t���[�h�~*/
}

hl2:for (i = 0; i<W4r; i++) {
	out[j][i] = exp(in[j][i] - max) / (s + c);/*0���Z�h�~*/
}
}
}

void s_l_bw(double x[Xc][W4r], uint8 T[Xc][W4r], int8 dx[Xc][W4r]) {
	/*�\�t�g�}�b�N�X�֐��ƌ����G���g���s�[�덷�֐��̋t�`�d���s���֐�*/
	uint4 j, i;
L1:for (i = 0; i<Xc; i++)
	L2 : for (j = 0; j<W4r; j++)
	dx[i][j] = 127 * (x[i][j] - T[i][j]);/*�����l���琮���l�ɕϊ�*/
   return;
}

/*bfnn:2�l���S�����j���[�����l�b�g���[�N(BFNN)�̏��`�d���s���֐�*/
void bfnn1(uint8 in[Xc][Xr], uint8 weight[W1c][W1r], int32 bias[W1r], int17 out[Xc][W1r]) {
	uint9 ib, id, ie;
L1:for (ie = 0; ie < Xc; ++ie)
{
L2:for (ib = 0; ib < W1r; ++ib)
{
	int13 sum = 0;
L3:for (id = 0; id < W1c; ++id)
{
	sum += popcnt_8(~(in[ie][id] ^ weight[id][ib])) << 1;
}
   out[ie][ib] = sum - Xr * 8 + bias[ib];
}
}
   return;
}

void bfnn2(uint8 in[Xc][W2c], uint8 weight[W2c][W2r], int32 bias[W2r], int17 out[Xc][W2r]) {
	uint7 ib, id, ie;
L1:for (ie = 0; ie < Xc; ++ie)
{
L2:for (ib = 0; ib < W2r; ++ib)
{
	int12 sum = 0;
L3:for (id = 0; id < W2c; ++id)
{
	sum += popcnt_8(~(in[ie][id] ^ weight[id][ib])) << 1;
}
   out[ie][ib] = sum - W2c * 8 + bias[ib];
}
}
   return;
}

void bfnn3(uint8 in[Xc][W3c], uint8 weight[W3c][W3r], int32 bias[W3r], int17 out[Xc][W3r]) {
	uint8 ib, id, ie;
L1:for (ie = 0; ie < Xc; ++ie)
{
L2:for (ib = 0; ib < W3r; ++ib)
{
	int12 sum = 0;
L3:for (id = 0; id < W3c; ++id)
{
	sum += popcnt_8(~(in[ie][id] ^ weight[id][ib])) << 1;
}
   out[ie][ib] = sum - W3c * 8 + bias[ib];
}
}
   return;
}

void bfnn4(uint8 in[Xc][W4c], uint8 weight[W4c][W4r], int32 bias[W4r], int17 out[Xc][W4r]) {
	uint8 ib, id, ie;
L1:for (ie = 0; ie < Xc; ++ie)
{
L2:for (ib = 0; ib < W4r; ++ib)
{
	int12 sum = 0;
L3:for (id = 0; id < W4c; ++id)
{
	sum += popcnt_8(~(in[ie][id] ^ weight[id][ib])) << 1;
}
   out[ie][ib] = sum - W4c * 8 + bias[ib];
}
}
   return;
}

/*bfnn_db:2�l���S�����j���[�����l�b�g���[�N(BFNN)�̃o�C�A�X�̌��z�Z�o���s���֐�*/
void bfnn4_db(int8 dy[Xc][W4r], int32 db[W4r]) {
	uint4 ib, ic;
dbL1:for (ib = 0; ib < W4r; ++ib)
{
	int16 sum = 0;
dbL2:for (ic = 0; ic<Xc; ic++) {
	sum += dy[ic][ib];
}
	 /*�o�C�A�X���X�V*/
	 db[ib] = ((db[ib] - sum / bl4)<8388607 && (db[ib] - sum / bl4)>-8388608) ? (db[ib] - sum / bl4) : ((db[ib] - sum / bl4)>0 ? 8388607 : -8388608);
}
}

void bfnn3_db(int32 dy[Xc][W3r], int32 db[W3r]) {
	uint7 ib, ic;
dbL1:for (ib = 0; ib < W3r; ++ib)
{
	int16 sum = 0;
dbL2:for (ic = 0; ic<Xc; ic++) {
	sum += dy[ic][ib];
}
	 /*�o�C�A�X���X�V*/
	 db[ib] = ((db[ib] - sum / bl3)<8388607 && (db[ib] - sum / bl3)>-8388608) ? (db[ib] - sum / bl3) : ((db[ib] - sum / bl3)>0 ? 8388607 : -8388608);
}
}

void bfnn2_db(int32 dy[Xc][W2r], int32 db[W2r]) {
	uint7 ib, ic;
dbL1:for (ib = 0; ib < W2r; ++ib)
{
	int32 sum = 0;
dbL2:for (ic = 0; ic<Xc; ic++) {
	sum += dy[ic][ib];
}
	 /*�o�C�A�X���X�V*/
	 db[ib] = ((db[ib] - sum / bl2)<8388607 && (db[ib] - sum / bl2)>-8388608) ? (db[ib] - sum / bl2) : ((db[ib] - sum / bl2)>0 ? 8388607 : -8388608);
}
}

void bfnn1_db(int32 dy[Xc][W1r], int32 db[W1r]) {
	uint9 ib, ic;
dbL1:for (ib = 0; ib < W1r; ++ib)
{
	int16 sum = 0;
dbL2:for (ic = 0; ic<Xc; ic++) {
	sum += dy[ic][ib];
}
	 /*�o�C�A�X���X�V*/
	 db[ib] = ((db[ib] - sum / bl1)<8388607 && (db[ib] - sum / bl1)>-8388608) ? (db[ib] - sum / bl1) : ((db[ib] - sum / bl1)>0 ? 8388607 : -8388608);
}
}

/*bfnn_dw:2�l���S�����j���[�����l�b�g���[�N(BFNN)�̏d�݌W���̌��z�Z�o���s���֐�*/
void bfnn4_dw(uint8 x[Xc][W4c], int8 dy[Xc][W4r], int8 dw[W4c * 8][W4r]) {
	uint5 ia, ib, ic, id;
dwL1:for (ia = 0; ia < W4c; ++ia) {
dwL2:for (ic = 0; ic < 8; ic++) {
dwL3:for (ib = 0; ib < W4r; ++ib) {
	int32 sum = 0;
dwL4:for (id = 0; id<Xc; id++) {
	sum += (((x[id][ia] & (0x80 >> ic)) > 0 ? 1 : -1)*dy[id][ib]);
}
 	 /*8bit���x�̏d�݌W�����X�V*/
	 dw[ia * 8 + ic][ib] = ((dw[ia * 8 + ic][ib] - sum / bl4)<127 && (dw[ia * 8 + ic][ib] - sum / bl4)>-128) ? dw[ia * 8 + ic][ib] - sum / bl4 : ((dw[ia * 8 + ic][ib] - sum / bl4)>0 ? 127 : -128);
}
}
}
}

void bfnn3_dw(uint8 x[Xc][W3c], int32 dy[Xc][W3r], int8 dw[W3c * 8][W3r]) {
	uint7 ia, ib, ic, id;
dwL1:for (ia = 0; ia < W3c; ++ia) {
dwL2:for (ic = 0; ic < 8; ic++) {
dwL3:for (ib = 0; ib < W3r; ++ib) {
	int32 sum = 0;
dwL4:for (id = 0; id<Xc; id++) {
	sum += (((x[id][ia] & (0x80 >> ic)) > 0 ? 1 : -1)*dy[id][ib]);
}
	 /*8bit���x�̏d�݌W�����X�V*/
	 dw[ia * 8 + ic][ib] = ((dw[ia * 8 + ic][ib] - sum / bl3)<127 && (dw[ia * 8 + ic][ib] - sum / bl3)>-128) ? dw[ia * 8 + ic][ib] - sum / bl3 : ((dw[ia * 8 + ic][ib] - sum / bl3)>0 ? 127 : -128);
}
}
}
}

void bfnn2_dw(uint8 x[Xc][W2c], int32 dy[Xc][W2r], int8 dw[W2c * 8][W2r]) {
	uint7 ia, ib, ic, id;
dwL1:for (ia = 0; ia < W2c; ++ia) {
dwL2:for (ic = 0; ic < 8; ++ic) {
dwL3:for (ib = 0; ib < W2r; ++ib) {
	int32 sum = 0;
dwL4:for (id = 0; id<Xc; id++) {
	sum += (((x[id][ia] & (0x80 >> ic)) > 0 ? 1 : -1)*dy[id][ib]);
}
	 /*8bit���x�̏d�݌W�����X�V*/
	 dw[ia * 8 + ic][ib] = ((dw[ia * 8 + ic][ib] - sum / bl2)<127 && (dw[ia * 8 + ic][ib] - sum / bl2)>-128) ? dw[ia * 8 + ic][ib] - sum / bl2 : ((dw[ia * 8 + ic][ib] - sum / bl2)>0 ? 127 : -128);

}
}
}
}

void bfnn1_dw(uint8 x[Xc][W1c], int32 dy[Xc][W1r], int8 dw[W1c * 8][W1r]) {
	uint9 ia, ib, ic, id;
dwL1:for (ia = 0; ia < W1c; ++ia) {
dwL2:for (ic = 0; ic < 8; ++ic) {
dwL3:for (ib = 0; ib < W1r; ++ib) {
	int32 sum = 0;
dwL4:for (id = 0; id<Xc; id++) {
	sum += (((x[id][ia] & (0x80 >> ic)) > 0 ? 1 : -1)*dy[id][ib]);
}
	 /*8bit���x�̏d�݌W�����X�V*/
	 dw[ia * 8 + ic][ib] = ((dw[ia * 8 + ic][ib] - sum / bl1)< 127 && (dw[ia * 8 + ic][ib] - sum / bl1)>-128) ? dw[ia * 8 + ic][ib] - sum / bl1 : ((dw[ia * 8 + ic][ib] - sum / bl1)>0 ? 127 : -128);

}
}
}
}

/*bfnn_dx:2�l���S�����j���[�����l�b�g���[�N(BFNN)�̓��͂̌��z�Z�o���s���֐�*/
void bfnn4_dx(uint8 w[W4c][W4r], int8 dy[Xc][W4r], int32 dx[Xc][W4c * 8]) {
	uint5 ib, id, ic, ia;
dxL1:for (ia = 0; ia<Xc; ia++) {
dxL2:for (ib = 0; ib < W4c; ++ib) {
dxL3:for (ic = 0; ic<8; ++ic) {
	int32 sum = 0;
dxL4:for (id = 0; id < W4r; ++id) {
	sum += (dy[ia][id] * ((w[ib][id] & (0x80 >> ic)) > 0 ? 1 : -1));
}
	 dx[ia][ib * 8 + ic] = sum;
}
}
}
}

void bfnn3_dx(uint8 w[W3c][W3r], int32 dy[Xc][W3r], int32 dx[Xc][W3c * 8]) {
	uint7 ib, id, ic, ia;
dxL1:for (ia = 0; ia<Xc; ia++) {
dxL2:for (ib = 0; ib < W3c; ++ib) {
dxL3:for (ic = 0; ic<8; ++ic) {
	int32 sum = 0;
dxL4:for (id = 0; id < W3r; ++id) {
	sum += (dy[ia][id] * ((w[ib][id] & (0x80 >> ic)) > 0 ? 1 : -1));
}
	 dx[ia][ib * 8 + ic] = sum;
}
}
}
}

void bfnn2_dx(uint8 w[W2c][W2r], int32 dy[Xc][W2r], int32 dx[Xc][W2c * 8]) {
	uint7 ib, id, ic, ia;
dxL1:for (ia = 0; ia<Xc; ia++) {
dxL2:for (ib = 0; ib < W2c; ++ib) {
dxL3:for (ic = 0; ic<8; ic++) {
	int32 sum = 0;
dxL4:for (id = 0; id < W2r; ++id) {
	sum += (dy[ia][id] * ((w[ib][id] & (0x80 >> ic)) > 0 ? 1 : -1));
}
	 dx[ia][ib * 8 + ic] = sum;
}
}
}
}

/*bfnn_bw:BFNN�̋t�`�d���s���֐�*/
void bfnn4_bw(int8 dy[Xc][W4r], uint8 x[Xc][W4c], uint8 w[W4c][W4r],
	int32 dx[Xc][W4c * 8], int8 dw[W4c * 8][W4r], int32 db[W4r])
{
	bfnn4_db(dy, db);
	bfnn4_dw(x, dy, dw);
	bfnn4_dx(w, dy, dx);
}

void bfnn3_bw(int32 dy[Xc][W3r], uint8 x[Xc][W3c], uint8 w[W3c][W3r],
	int32 dx[Xc][W3c * 8], int8 dw[W3c * 8][W3r], int32 db[W3r])
{
	bfnn3_db(dy, db);
	bfnn3_dw(x, dy, dw);
	bfnn3_dx(w, dy, dx);
}

void bfnn2_bw(int32 dy[Xc][W3c * 8], uint8 x[Xc][W2c], uint8 w[W2c][W2r],
	int32 dx[Xc][W2c * 8], int8 dw[W2c * 8][W2r], int32 db[W2r]) {
	bfnn2_db(dy, db);
	bfnn2_dw(x, dy, dw);
	bfnn2_dx(w, dy, dx);
}

void bfnn1_bw(int32 dy[Xc][W2c * 8], uint8 x[Xc][W1c],
	int8 dw[W1c * 8][W1r], int32 db[W1r]) {
	bfnn1_db(dy, db);
	bfnn1_dw(x, dy, dw);
}

/*BNN���f��Ver.1_0�̐��_���s���֐�*/
void model_v_1(uint8 in[Xc][Xr],
	uint8 weight1[W1c][W1r], int32 bias1[W1r],
	uint8 weight2[W2c][W2r], int32 bias2[W2r],
	uint8 weight3[W3c][W3r], int32 bias3[W3r],
	uint8 weight4[W4c][W4r], int32 bias4[W4r],
	double out[Xc][W4r],
	int17 out1[Xc][W1r], int17 out2[Xc][W2r], int17 out3[Xc][W3r], int17 out4[Xc][W4r],
	uint8 in2[Xc][W2c], uint8 in3[Xc][W3c], uint8 in4[Xc][W4c])
{

	bfnn1(in, weight1, bias1, out1);
	b_a1(out1, in2);
	bfnn2(in2, weight2, bias2, out2);
	b_a2(out2, in3);
	bfnn3(in3, weight3, bias3, out3);
	b_a3(out3, in4);
	bfnn4(in4, weight4, bias4, out4);
	sf(out4, out);

}

/*BNN���f��Ver.1_0�̊w�K/���_���s���֐�*/
void learn_model_v_1(uint8 in[Xc][Xr],
	uint8 weight1[W1c][W1r], int32 bias1[W1r],
	uint8 weight2[W2c][W2r], int32 bias2[W2r],
	uint8 weight3[W3c][W3r], int32 bias3[W3r],
	uint8 weight4[W4c][W4r], int32 bias4[W4r],
	double out[Xc][W4r],
	uint8 T[Xc][W4r],/*�������x��*/
	uint8 k          /*�w�K�t���O*/
) {
	int8 dw1[W1c * 8][W1r], dw2[W2c * 8][W2r], dw3[W3c * 8][W3r], dw4[W4c * 8][W4r];
	uint8 in2[Xc][W2c], in3[Xc][W3c], in4[Xc][W4c]; int8 ds[Xc][W4r];
	int17 out1[Xc][W1r], out2[Xc][W2r], out3[Xc][W3r], out4[Xc][W4r];
	int32 df4[Xc][W4c * 8], df3[Xc][W3c * 8], df2[Xc][W2c * 8];

	model_v_1(in, weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4, out, out1, out2, out3,out4, in2, in3, in4);
	if (k == 0) {
		s_l_bw(out, T, ds);
		bfnn4_bw(ds, in4, weight4, df4, dw4, bias4);
		b_a3_bw(out3, df4);
		bfnn3_bw(df4, in3, weight3, df3, dw3, bias3);
		b_a2_bw(out2, df3);
		bfnn2_bw(df3, in2, weight2, df2, dw2, bias2);
		b_a1_bw(out1, df2);
		bfnn1_bw(df2, in, dw1, bias1);

		uint10 ia, ib, id, ic;
		/*�X�V����8bit���x�̏d�݌W�����Q�l�������*/
	dw4L1:for (ia = 0; ia < W4r; ++ia) {
	dw4L2:for (ib = 0; ib < W4c; ++ib) {
		weight4[ib][ia] = (dw4[ib * 8][ia]>0 ? 0x80 : 0x00) |
			(dw4[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
			(dw4[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
			(dw4[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
			(dw4[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
			(dw4[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
			(dw4[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
			(dw4[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
	}
	}

	dw3L1:for (ia = 0; ia < W3r; ++ia) {
	dw3L2:for (ib = 0; ib < W3c; ++ib) {
		weight3[ib][ia] = (dw3[ib * 8][ia]>0 ? 0x80 : 0x00) |
			(dw3[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
			(dw3[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
			(dw3[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
			(dw3[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
			(dw3[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
			(dw3[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
			(dw3[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
	}
	}
	  dw2L1:for (ia = 0; ia < W2r; ++ia) {
	  dw2L2:for (ib = 0; ib < W2c; ++ib) {
		  weight2[ib][ia] = (dw2[ib * 8][ia]>0 ? 0x80 : 0x00) |
			  (dw2[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
			  (dw2[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
			  (dw2[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
			  (dw2[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
			  (dw2[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
			  (dw2[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
			  (dw2[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
	  }
	  }
		dw1L1:for (ia = 0; ia < W1r; ++ia) {
		dw1L2:for (ib = 0; ib < W1c; ++ib) {
			weight1[ib][ia] = (dw1[ib * 8][ia]>0 ? 0x80 : 0x00) |
				(dw1[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
				(dw1[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
				(dw1[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
				(dw1[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
				(dw1[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
				(dw1[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
				(dw1[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
		}
		}
	}
	return;
}

/*BNN���f��Ver.2_0�̐��_���s���֐�*/
void model_v_2(uint8 in[Xc][Xr],
	uint8 weight1[W1c][W1r], int32 bias1[W1r],
	uint8 weight2[W2c][W2r], int32 bias2[W2r],
	uint8 weight3[W3c][W3r], int32 bias3[W3r],
	uint8 weight4[W4c][W4r], int32 bias4[W4r],
	double out[Xc][W4r],
	int17 out1[Xc][W1r], int17 out2[Xc][W2r], int17 out3[Xc][W3r], int17 out4[Xc][W4r],
	int17 bo1[Xc][W1r], int17 bo2[Xc][W2r], int17 bo3[Xc][W3r],
	uint8 in2[Xc][W2c], uint8 in3[Xc][W3c], uint8 in4[Xc][W4c])
{

	bfnn1(in, weight1, bias1, out1);
	bd1(out1, bo1);
	b_a1(bo1, in2);
	bfnn2(in2, weight2, bias2, out2);
	bd2(out2, bo2);
	b_a2(bo2, in3);
	bfnn3(in3, weight3, bias3, out3);
	bd3(out3, bo3);
	b_a3(bo3, in4);
	bfnn4(in4, weight4, bias4, out4);
	sf(out4, out);

}

/*BNN���f��Ver.2_0�̊w�K/���_���s���֐�*/
void learn_model_v_2(uint8 in[Xc][Xr],
	uint8 weight1[W1c][W1r], int32 bias1[W1r],
	uint8 weight2[W2c][W2r], int32 bias2[W2r],
	uint8 weight3[W3c][W3r], int32 bias3[W3r],
	uint8 weight4[W4c][W4r], int32 bias4[W4r],
	double out[Xc][W4r],
	uint8 T[Xc][W4r],/*�������x��*/
	uint8 k          /*�w�K�t���O*/
) {
	int8 dw1[W1c * 8][W1r], dw2[W2c * 8][W2r], dw3[W3c * 8][W3r], dw4[W4c * 8][W4r];
	uint8 in2[Xc][W2c], in3[Xc][W3c], in4[Xc][W4c]; int8 ds[Xc][W4r];
	int17 out1[Xc][W1r], out2[Xc][W2r], out3[Xc][W3r], out4[Xc][W4r];
	int17 bo1[Xc][W1r],bo2[Xc][W2r], bo3[Xc][W3r];
	int32 df4[Xc][W4c * 8],df3[Xc][W3c * 8], df2[Xc][W2c * 8];

	model_v_2(in, weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4, out, out1, out2, out3, out4, bo1, bo2, bo3,in2, in3, in4);
	if (k == 0) {
		s_l_bw(out, T, ds);
		bfnn4_bw(ds, in4, weight4, df4, dw4, bias4);
		leaky_relu3_bw(bo3,df4);
		bd3_bw(df4);
		bfnn3_bw(df4, in3, weight3, df3, dw3, bias3);
		leaky_relu2_bw(bo2,df3);
		bd2_bw(df3);
		bfnn2_bw(df3, in2, weight2, df2, dw2, bias2);
		leaky_relu1_bw(bo1,df2);
		bd1_bw(df2);
		bfnn1_bw(df2, in, dw1, bias1);

		uint10 ia, ib, id, ic;
		/*�X�V����8bit���x�̏d�݌W�����Q�l�������*/
	dw4L1:for (ia = 0; ia < W4r; ++ia) {
	dw4L2:for (ib = 0; ib < W4c; ++ib) {
		weight4[ib][ia] = (dw4[ib * 8][ia]>0 ? 0x80 : 0x00) |
			(dw4[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
			(dw4[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
			(dw4[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
			(dw4[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
			(dw4[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
			(dw4[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
			(dw4[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
	}
	}

	dw3L1:for (ia = 0; ia < W3r; ++ia) {
	dw3L2:for (ib = 0; ib < W3c; ++ib) {
		weight3[ib][ia] = (dw3[ib * 8][ia]>0 ? 0x80 : 0x00) |
			(dw3[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
			(dw3[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
			(dw3[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
			(dw3[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
			(dw3[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
			(dw3[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
			(dw3[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
	}
	}
	  dw2L1:for (ia = 0; ia < W2r; ++ia) {
	  dw2L2:for (ib = 0; ib < W2c; ++ib) {
		  weight2[ib][ia] = (dw2[ib * 8][ia]>0 ? 0x80 : 0x00) |
			  (dw2[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
			  (dw2[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
			  (dw2[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
			  (dw2[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
			  (dw2[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
			  (dw2[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
			  (dw2[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
	  }
	  }
		dw1L1:for (ia = 0; ia < W1r; ++ia) {
		dw1L2:for (ib = 0; ib < W1c; ++ib) {
			weight1[ib][ia] = (dw1[ib * 8][ia]>0 ? 0x80 : 0x00) |
				(dw1[(ib * 8) + 1][ia]>0 ? 0x40 : 0x00) |
				(dw1[(ib * 8) + 2][ia]>0 ? 0x20 : 0x00) |
				(dw1[(ib * 8) + 3][ia]>0 ? 0x10 : 0x00) |
				(dw1[(ib * 8) + 4][ia]>0 ? 0x08 : 0x00) |
				(dw1[(ib * 8) + 5][ia]>0 ? 0x04 : 0x00) |
				(dw1[(ib * 8) + 6][ia]>0 ? 0x02 : 0x00) |
				(dw1[(ib * 8) + 7][ia]>0 ? 0x01 : 0x00);
		}
		}
	}
	return;
}
