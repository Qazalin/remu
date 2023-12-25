
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3_2_9_5_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s3, s13, 0x66666667                           // 000000001714: 9703FF0D 66666667
	s_mul_i32 s2, s15, 0x4d                                    // 00000000171C: 9602FF0F 0000004D
	s_lshr_b32 s8, s3, 31                                      // 000000001724: 85089F03
	s_ashr_i32 s9, s3, 1                                       // 000000001728: 86098103
	s_ashr_i32 s3, s2, 31                                      // 00000000172C: 86039F02
	s_add_i32 s9, s9, s8                                       // 000000001730: 81090809
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001734: 84828202
	s_mul_i32 s8, s9, 5                                        // 000000001738: 96088509
	s_mul_i32 s10, s9, 7                                       // 00000000173C: 960A8709
	s_sub_i32 s12, s13, s8                                     // 000000001740: 818C080D
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001748: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000174C: 82070307
	s_ashr_i32 s11, s10, 31                                    // 000000001750: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001754: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001758: 8482820A
	s_add_u32 s9, s6, s2                                       // 00000000175C: 80090206
	s_addc_u32 s7, s7, s3                                      // 000000001760: 82070307
	s_ashr_i32 s13, s12, 31                                    // 000000001764: 860D9F0C
	s_mul_i32 s6, s15, 18                                      // 000000001768: 9606920F
	s_lshl_b64 s[2:3], s[12:13], 2                             // 00000000176C: 8482820C
	s_mul_i32 s12, s14, 9                                      // 000000001770: 960C890E
	s_add_u32 s10, s9, s2                                      // 000000001774: 800A0209
	s_addc_u32 s11, s7, s3                                     // 000000001778: 820B0307
	s_ashr_i32 s7, s6, 31                                      // 00000000177C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001780: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001784: 84868206
	s_add_u32 s6, s0, s6                                       // 000000001788: 80060600
	s_addc_u32 s7, s1, s7                                      // 00000000178C: 82070701
	s_ashr_i32 s13, s12, 31                                    // 000000001790: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001794: BF870499
	s_lshl_b64 s[0:1], s[12:13], 2                             // 000000001798: 8480820C
	s_add_u32 s0, s6, s0                                       // 00000000179C: 80000006
	s_addc_u32 s1, s7, s1                                      // 0000000017A0: 82010107
	s_load_b256 s[16:23], s[0:1], null                         // 0000000017A4: F40C0400 F8000000
	s_clause 0x3                                               // 0000000017AC: BF850003
	s_load_b64 s[6:7], s[10:11], null                          // 0000000017B0: F4040185 F8000000
	s_load_b32 s9, s[10:11], 0x8                               // 0000000017B8: F4000245 F8000008
	s_load_b64 s[12:13], s[10:11], 0x1c                        // 0000000017C0: F4040305 F800001C
	s_load_b32 s24, s[10:11], 0x24                             // 0000000017C8: F4000605 F8000024
	s_waitcnt lgkmcnt(0)                                       // 0000000017D0: BF89FC07
	v_fma_f32 v0, s6, s16, 0                                   // 0000000017D4: D6130000 02002006
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017DC: BF8700A1
	v_fmac_f32_e64 v0, s7, s17                                 // 0000000017E0: D52B0000 00002207
	s_load_b64 s[6:7], s[10:11], 0x38                          // 0000000017E8: F4040185 F8000038
	v_fmac_f32_e64 v0, s9, s18                                 // 0000000017F0: D52B0000 00002409
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000017F8: BF870001
	v_fmac_f32_e64 v0, s12, s19                                // 0000000017FC: D52B0000 0000260C
	s_load_b32 s9, s[10:11], 0x40                              // 000000001804: F4000245 F8000040
	s_load_b32 s12, s[0:1], 0x20                               // 00000000180C: F4000300 F8000020
	s_mul_i32 s0, s15, 0x5a                                    // 000000001814: 9600FF0F 0000005A
	s_mul_i32 s10, s14, 45                                     // 00000000181C: 960AAD0E
	s_ashr_i32 s1, s0, 31                                      // 000000001820: 86019F00
	v_fmac_f32_e64 v0, s13, s20                                // 000000001824: D52B0000 0000280D
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000182C: 84808200
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001830: BF8700A9
	s_add_u32 s4, s4, s0                                       // 000000001834: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001838: 82050105
	v_fmac_f32_e64 v0, s24, s21                                // 00000000183C: D52B0000 00002A18
	s_ashr_i32 s11, s10, 31                                    // 000000001844: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001848: BF8700A9
	s_lshl_b64 s[0:1], s[10:11], 2                             // 00000000184C: 8480820A
	s_waitcnt lgkmcnt(0)                                       // 000000001850: BF89FC07
	v_fmac_f32_e64 v0, s6, s22                                 // 000000001854: D52B0000 00002C06
	s_add_u32 s4, s4, s0                                       // 00000000185C: 80040004
	s_addc_u32 s5, s5, s1                                      // 000000001860: 82050105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001864: BF870091
	v_fmac_f32_e64 v0, s7, s23                                 // 000000001868: D52B0000 00002E07
	v_fmac_f32_e64 v0, s9, s12                                 // 000000001870: D52B0000 00001809
	s_ashr_i32 s9, s8, 31                                      // 000000001878: 86099F08
	v_mov_b32_e32 v1, 0                                        // 00000000187C: 7E020280
	s_lshl_b64 s[0:1], s[8:9], 2                               // 000000001880: 84808208
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001884: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001888: 20000080
	s_add_u32 s0, s4, s0                                       // 00000000188C: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001890: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001894: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001898: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 00000000189C: DC6A0000 00000001
	s_nop 0                                                    // 0000000018A4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018A8: BFB60003
	s_endpgm                                                   // 0000000018AC: BFB00000
