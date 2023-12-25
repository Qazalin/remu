
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_10_3_2_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x55555556                           // 000000001714: 9708FF0D 55555556
	s_mul_i32 s2, s15, 0x4d                                    // 00000000171C: 9602FF0F 0000004D
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
	s_mul_i32 s2, s14, 10                                      // 000000001764: 96028A0E
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
	s_load_b128 s[8:11], s[24:25], 0x1c                        // 0000000017A8: F408020C F800001C
	s_waitcnt lgkmcnt(0)                                       // 0000000017B0: BF89FC07
	v_fma_f32 v0, s0, s16, 0                                   // 0000000017B4: D6130000 02002000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017BC: BF8700A1
	v_fmac_f32_e64 v0, s1, s17                                 // 0000000017C0: D52B0000 00002201
	s_load_b64 s[0:1], s[26:27], 0x20                          // 0000000017C8: F404000D F8000020
	v_fmac_f32_e64 v0, s2, s18                                 // 0000000017D0: D52B0000 00002402
	s_mul_i32 s2, s15, 0xb4                                    // 0000000017D8: 9602FF0F 000000B4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000017E0: BF8704A1
	v_fmac_f32_e64 v0, s3, s19                                 // 0000000017E4: D52B0000 00002603
	s_ashr_i32 s3, s2, 31                                      // 0000000017EC: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000017F0: 84828202
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017F4: BF8700C1
	v_fmac_f32_e64 v0, s13, s20                                // 0000000017F8: D52B0000 0000280D
	s_load_b32 s13, s[24:25], 0x2c                             // 000000001800: F400034C F800002C
	s_add_u32 s4, s4, s2                                       // 000000001808: 80040204
	s_addc_u32 s5, s5, s3                                      // 00000000180C: 82050305
	v_fmac_f32_e64 v0, s8, s21                                 // 000000001810: D52B0000 00002A08
	s_mul_i32 s8, s14, 30                                      // 000000001818: 96089E0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 00000000181C: BF8704A1
	v_fmac_f32_e64 v0, s9, s22                                 // 000000001820: D52B0000 00002C09
	s_ashr_i32 s9, s8, 31                                      // 000000001828: 86099F08
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000182C: 84828208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001830: BF8700C1
	v_fmac_f32_e64 v0, s10, s23                                // 000000001834: D52B0000 00002E0A
	s_add_u32 s2, s4, s2                                       // 00000000183C: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001840: 82030305
	s_waitcnt lgkmcnt(0)                                       // 000000001844: BF89FC07
	v_fmac_f32_e64 v0, s11, s0                                 // 000000001848: D52B0000 0000000B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001850: BF870141
	v_fmac_f32_e64 v0, s13, s1                                 // 000000001854: D52B0000 0000020D
	s_ashr_i32 s13, s12, 31                                    // 00000000185C: 860D9F0C
	v_mov_b32_e32 v1, 0                                        // 000000001860: 7E020280
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001864: 8480820C
	v_max_f32_e32 v0, 0, v0                                    // 000000001868: 20000080
	s_add_u32 s0, s2, s0                                       // 00000000186C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001870: 82010103
	s_add_u32 s0, s0, s6                                       // 000000001874: 80000600
	s_addc_u32 s1, s1, s7                                      // 000000001878: 82010701
	global_store_b32 v1, v0, s[0:1]                            // 00000000187C: DC6A0000 00000001
	s_nop 0                                                    // 000000001884: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001888: BFB60003
	s_endpgm                                                   // 00000000188C: BFB00000
