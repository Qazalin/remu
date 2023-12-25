
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_11_3_3_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x55555556                           // 000000001714: 9708FF0D 55555556
	s_mul_i32 s2, s15, 0xe7                                    // 00000000171C: 9602FF0F 000000E7
	s_lshr_b32 s9, s8, 31                                      // 000000001724: 85099F08
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_add_i32 s8, s8, s9                                       // 00000000172C: 81080908
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_mul_i32 s12, s8, 3                                       // 000000001734: 960C8308
	s_mul_i32 s8, s8, 7                                        // 000000001738: 96088708
	s_sub_i32 s10, s13, s12                                    // 00000000173C: 818A0C0D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001744: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001748: 82070307
	s_ashr_i32 s9, s8, 31                                      // 00000000174C: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[8:9], 2                               // 000000001754: 84828208
	s_add_u32 s8, s6, s2                                       // 000000001758: 80080206
	s_addc_u32 s3, s7, s3                                      // 00000000175C: 82030307
	s_ashr_i32 s11, s10, 31                                    // 000000001760: 860B9F0A
	s_mul_i32 s2, s14, 15                                      // 000000001764: 96028F0E
	s_lshl_b64 s[6:7], s[10:11], 2                             // 000000001768: 8486820A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000176C: BF8704B9
	s_add_u32 s24, s8, s6                                      // 000000001770: 80180608
	s_addc_u32 s25, s3, s7                                     // 000000001774: 82190703
	s_ashr_i32 s3, s2, 31                                      // 000000001778: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000177C: 84828202
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001780: BF870009
	s_add_u32 s26, s0, s2                                      // 000000001784: 801A0200
	s_addc_u32 s27, s1, s3                                     // 000000001788: 821B0301
	s_load_b256 s[16:23], s[26:27], null                       // 00000000178C: F40C040D F8000000
	s_clause 0x2                                               // 000000001794: BF850002
	s_load_b128 s[0:3], s[24:25], null                         // 000000001798: F408000C F8000000
	s_load_b32 s13, s[24:25], 0x10                             // 0000000017A0: F400034C F8000010
	s_load_b128 s[8:11], s[24:25], 0x134                       // 0000000017A8: F408020C F8000134
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_fma_f32 v0, s0, s16, 0                                   // 0000000017B4: D6130000 02002000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017BC: BF870091
	v_fmac_f32_e64 v0, s1, s17                                 // 0000000017C0: D52B0000 00002201
	v_fmac_f32_e64 v0, s2, s18                                 // 0000000017C8: D52B0000 00002402
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 0000000017D0: BF8700B1
	v_fmac_f32_e64 v0, s3, s19                                 // 0000000017D4: D52B0000 00002603
	s_load_b128 s[0:3], s[26:27], 0x20                         // 0000000017DC: F408000D F8000020
	s_load_b128 s[16:19], s[24:25], 0x268                      // 0000000017E4: F408040C F8000268
	v_fmac_f32_e64 v0, s13, s20                                // 0000000017EC: D52B0000 0000280D
	s_load_b32 s13, s[24:25], 0x144                            // 0000000017F4: F400034C F8000144
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017FC: BF870091
	v_fmac_f32_e64 v0, s8, s21                                 // 000000001800: D52B0000 00002A08
	v_fmac_f32_e64 v0, s9, s22                                 // 000000001808: D52B0000 00002C09
	s_load_b64 s[8:9], s[26:27], 0x30                          // 000000001810: F404020D F8000030
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001818: BF8700A1
	v_fmac_f32_e64 v0, s10, s23                                // 00000000181C: D52B0000 00002E0A
	s_waitcnt lgkmcnt(0)                                       // 000000001824: BF89FC07
	v_fmac_f32_e64 v0, s11, s0                                 // 000000001828: D52B0000 0000000B
	s_load_b32 s10, s[24:25], 0x278                            // 000000001830: F400028C F8000278
	s_load_b32 s11, s[26:27], 0x38                             // 000000001838: F40002CD F8000038
	s_mul_i32 s0, s15, 0xc6                                    // 000000001840: 9600FF0F 000000C6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001848: BF8704A1
	v_fmac_f32_e64 v0, s13, s1                                 // 00000000184C: D52B0000 0000020D
	s_ashr_i32 s1, s0, 31                                      // 000000001854: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001858: 84808200
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000185C: BF8700C1
	v_fmac_f32_e64 v0, s16, s2                                 // 000000001860: D52B0000 00000410
	s_mul_i32 s2, s14, 33                                      // 000000001868: 9602A10E
	s_add_u32 s4, s4, s0                                       // 00000000186C: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001870: 82050105
	v_fmac_f32_e64 v0, s17, s3                                 // 000000001874: D52B0000 00000611
	s_ashr_i32 s3, s2, 31                                      // 00000000187C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001880: BF870099
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001884: 84808202
	v_fmac_f32_e64 v0, s18, s8                                 // 000000001888: D52B0000 00001012
	s_add_u32 s2, s4, s0                                       // 000000001890: 80020004
	s_addc_u32 s3, s5, s1                                      // 000000001894: 82030105
	s_ashr_i32 s13, s12, 31                                    // 000000001898: 860D9F0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000189C: BF8704A1
	v_fmac_f32_e64 v0, s19, s9                                 // 0000000018A0: D52B0000 00001213
	s_lshl_b64 s[0:1], s[12:13], 2                             // 0000000018A8: 8480820C
	s_add_u32 s0, s2, s0                                       // 0000000018AC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000018B0: 82010103
	s_waitcnt lgkmcnt(0)                                       // 0000000018B4: BF89FC07
	v_fmac_f32_e64 v0, s10, s11                                // 0000000018B8: D52B0000 0000160A
	v_mov_b32_e32 v1, 0                                        // 0000000018C0: 7E020280
	s_add_u32 s0, s0, s6                                       // 0000000018C4: 80000600
	s_addc_u32 s1, s1, s7                                      // 0000000018C8: 82010701
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000018CC: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000018D0: 20000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000018D4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018DC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018E0: BFB60003
	s_endpgm                                                   // 0000000018E4: BFB00000
