
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_3_9_26_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s3, s13, 0x4ec4ec4f                           // 000000001714: 9703FF0D 4EC4EC4F
	s_mul_i32 s2, s15, 0x39c                                   // 00000000171C: 9602FF0F 0000039C
	s_lshr_b32 s9, s3, 31                                      // 000000001724: 85099F03
	s_ashr_i32 s3, s3, 3                                       // 000000001728: 86038303
	s_mul_i32 s8, s14, 0x134                                   // 00000000172C: 9608FF0E 00000134
	s_add_i32 s11, s3, s9                                      // 000000001734: 810B0903
	s_ashr_i32 s3, s2, 31                                      // 000000001738: 86039F02
	s_mul_i32 s10, s11, 26                                     // 00000000173C: 960A9A0B
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001740: 84828202
	s_sub_i32 s12, s13, s10                                    // 000000001744: 818C0A0D
	s_waitcnt lgkmcnt(0)                                       // 000000001748: BF89FC07
	s_add_u32 s13, s6, s2                                      // 00000000174C: 800D0206
	s_addc_u32 s7, s7, s3                                      // 000000001750: 82070307
	s_ashr_i32 s9, s8, 31                                      // 000000001754: 86099F08
	s_mul_i32 s6, s11, 28                                      // 000000001758: 96069C0B
	s_lshl_b64 s[2:3], s[8:9], 2                               // 00000000175C: 84828208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001760: BF8704B9
	s_add_u32 s8, s13, s2                                      // 000000001764: 8008020D
	s_addc_u32 s9, s7, s3                                      // 000000001768: 82090307
	s_ashr_i32 s7, s6, 31                                      // 00000000176C: 86079F06
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001770: 84828206
	s_mul_i32 s6, s14, 9                                       // 000000001774: 9606890E
	s_add_u32 s7, s8, s2                                       // 000000001778: 80070208
	s_addc_u32 s9, s9, s3                                      // 00000000177C: 82090309
	s_ashr_i32 s13, s12, 31                                    // 000000001780: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001784: BF870499
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001788: 8482820C
	s_add_u32 s8, s7, s2                                       // 00000000178C: 80080207
	s_addc_u32 s9, s9, s3                                      // 000000001790: 82090309
	s_ashr_i32 s7, s6, 31                                      // 000000001794: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001798: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 00000000179C: 84868206
	s_add_u32 s0, s0, s6                                       // 0000000017A0: 80000600
	s_addc_u32 s1, s1, s7                                      // 0000000017A4: 82010701
	s_load_b256 s[16:23], s[0:1], null                         // 0000000017A8: F40C0400 F8000000
	s_clause 0x2                                               // 0000000017B0: BF850002
	s_load_b64 s[6:7], s[8:9], null                            // 0000000017B4: F4040184 F8000000
	s_load_b32 s11, s[8:9], 0x8                                // 0000000017BC: F40002C4 F8000008
	s_load_b64 s[12:13], s[8:9], 0x70                          // 0000000017C4: F4040304 F8000070
	s_waitcnt lgkmcnt(0)                                       // 0000000017CC: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 0000000017D0: D6130000 02002006
	s_load_b32 s16, s[8:9], 0x78                               // 0000000017D8: F4000404 F8000078
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E0: BF8700A1
	v_fmac_f32_e64 v0, s7, s17                                 // 0000000017E4: D52B0000 00002207
	s_load_b64 s[6:7], s[8:9], 0xe0                            // 0000000017EC: F4040184 F80000E0
	v_fmac_f32_e64 v0, s11, s18                                // 0000000017F4: D52B0000 0000240B
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017FC: BF870001
	v_fmac_f32_e64 v0, s12, s19                                // 000000001800: D52B0000 0000260C
	s_load_b32 s11, s[8:9], 0xe8                               // 000000001808: F40002C4 F80000E8
	s_load_b32 s12, s[0:1], 0x20                               // 000000001810: F4000300 F8000020
	s_mul_i32 s0, s15, 0x2be                                   // 000000001818: 9600FF0F 000002BE
	s_mul_i32 s8, s14, 0xea                                    // 000000001820: 9608FF0E 000000EA
	s_ashr_i32 s1, s0, 31                                      // 000000001828: 86019F00
	v_fmac_f32_e64 v0, s13, s20                                // 00000000182C: D52B0000 0000280D
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001834: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001838: BF8704D9
	s_add_u32 s4, s4, s0                                       // 00000000183C: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001840: 82050105
	s_waitcnt lgkmcnt(0)                                       // 000000001844: BF89FC07
	v_fmac_f32_e64 v0, s16, s21                                // 000000001848: D52B0000 00002A10
	s_ashr_i32 s9, s8, 31                                      // 000000001850: 86099F08
	s_lshl_b64 s[0:1], s[8:9], 2                               // 000000001854: 84808208
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001858: BF8700B1
	v_fmac_f32_e64 v0, s6, s22                                 // 00000000185C: D52B0000 00002C06
	s_add_u32 s4, s4, s0                                       // 000000001864: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001868: 82050105
	v_fmac_f32_e64 v0, s7, s23                                 // 00000000186C: D52B0000 00002E07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001874: BF870141
	v_fmac_f32_e64 v0, s11, s12                                // 000000001878: D52B0000 0000180B
	s_ashr_i32 s11, s10, 31                                    // 000000001880: 860B9F0A
	v_mov_b32_e32 v1, 0                                        // 000000001884: 7E020280
	s_lshl_b64 s[0:1], s[10:11], 2                             // 000000001888: 8480820A
	v_max_f32_e32 v0, 0, v0                                    // 00000000188C: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001890: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001894: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001898: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000189C: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 0000000018A0: DC6A0000 00000001
	s_nop 0                                                    // 0000000018A8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018AC: BFB60003
	s_endpgm                                                   // 0000000018B0: BFB00000
