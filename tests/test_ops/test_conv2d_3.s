
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_11_3_3_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 7                                       // 000000001714: 9608870E
	s_mov_b32 s2, s13                                          // 000000001718: BE82000D
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s8, s6, s8                                       // 00000000172C: 80080806
	s_addc_u32 s9, s7, s9                                      // 000000001730: 82090907
	s_ashr_i32 s3, s13, 31                                     // 000000001734: 86039F0D
	s_lshl_b64 s[6:7], s[2:3], 2                               // 000000001738: 84868202
	s_mul_i32 s2, s15, 15                                      // 00000000173C: 96028F0F
	s_add_u32 s12, s8, s6                                      // 000000001740: 800C0608
	s_addc_u32 s13, s9, s7                                     // 000000001744: 820D0709
	s_ashr_i32 s3, s2, 31                                      // 000000001748: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000174C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001750: 84828202
	s_add_u32 s24, s0, s2                                      // 000000001754: 80180200
	s_addc_u32 s25, s1, s3                                     // 000000001758: 82190301
	s_load_b256 s[16:23], s[24:25], null                       // 00000000175C: F40C040C F8000000
	s_clause 0x2                                               // 000000001764: BF850002
	s_load_b128 s[0:3], s[12:13], null                         // 000000001768: F4080006 F8000000
	s_load_b32 s26, s[12:13], 0x10                             // 000000001770: F4000686 F8000010
	s_load_b128 s[8:11], s[12:13], 0x134                       // 000000001778: F4080206 F8000134
	s_waitcnt lgkmcnt(0)                                       // 000000001780: BF89FC07
	v_fma_f32 v0, s0, s16, 0                                   // 000000001784: D6130000 02002000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000178C: BF870091
	v_fmac_f32_e64 v0, s1, s17                                 // 000000001790: D52B0000 00002201
	v_fmac_f32_e64 v0, s2, s18                                 // 000000001798: D52B0000 00002402
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000017A0: BF8700B1
	v_fmac_f32_e64 v0, s3, s19                                 // 0000000017A4: D52B0000 00002603
	s_load_b128 s[0:3], s[24:25], 0x20                         // 0000000017AC: F408000C F8000020
	s_load_b128 s[16:19], s[12:13], 0x268                      // 0000000017B4: F4080406 F8000268
	v_fmac_f32_e64 v0, s26, s20                                // 0000000017BC: D52B0000 0000281A
	s_load_b32 s20, s[12:13], 0x144                            // 0000000017C4: F4000506 F8000144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017CC: BF870091
	v_fmac_f32_e64 v0, s8, s21                                 // 0000000017D0: D52B0000 00002A08
	v_fmac_f32_e64 v0, s9, s22                                 // 0000000017D8: D52B0000 00002C09
	s_load_b64 s[8:9], s[24:25], 0x30                          // 0000000017E0: F404020C F8000030
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E8: BF8700A1
	v_fmac_f32_e64 v0, s10, s23                                // 0000000017EC: D52B0000 00002E0A
	s_waitcnt lgkmcnt(0)                                       // 0000000017F4: BF89FC07
	v_fmac_f32_e64 v0, s11, s0                                 // 0000000017F8: D52B0000 0000000B
	s_load_b32 s10, s[12:13], 0x278                            // 000000001800: F4000286 F8000278
	s_load_b32 s11, s[24:25], 0x38                             // 000000001808: F40002CC F8000038
	s_mul_i32 s0, s15, 33                                      // 000000001810: 9600A10F
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001814: BF8704A1
	v_fmac_f32_e64 v0, s20, s1                                 // 000000001818: D52B0000 00000214
	s_ashr_i32 s1, s0, 31                                      // 000000001820: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001824: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001828: BF8700C1
	v_fmac_f32_e64 v0, s16, s2                                 // 00000000182C: D52B0000 00000410
	s_mul_i32 s2, s14, 3                                       // 000000001834: 9602830E
	s_add_u32 s4, s4, s0                                       // 000000001838: 80040004
	s_addc_u32 s5, s5, s1                                      // 00000000183C: 82050105
	v_fmac_f32_e64 v0, s17, s3                                 // 000000001840: D52B0000 00000611
	s_ashr_i32 s3, s2, 31                                      // 000000001848: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870099
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001850: 84808202
	v_fmac_f32_e64 v0, s18, s8                                 // 000000001854: D52B0000 00001012
	s_add_u32 s0, s4, s0                                       // 00000000185C: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001860: 82010105
	s_add_u32 s0, s0, s6                                       // 000000001864: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001868: 82010701
	v_fmac_f32_e64 v0, s19, s9                                 // 00000000186C: D52B0000 00001213
	s_waitcnt lgkmcnt(0)                                       // 000000001874: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001878: BF870091
	v_fmac_f32_e64 v0, s10, s11                                // 00000000187C: D52B0000 0000160A
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001884: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 00000000188C: DC6A0000 00000001
	s_nop 0                                                    // 000000001894: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001898: BFB60003
	s_endpgm                                                   // 00000000189C: BFB00000
