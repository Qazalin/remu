
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_11_5_3>:
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
	s_mul_i32 s6, s14, 3                                       // 000000001768: 9606830E
	s_lshl_b64 s[2:3], s[12:13], 2                             // 00000000176C: 8482820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 000000001770: BF8704B9
	s_add_u32 s10, s9, s2                                      // 000000001774: 800A0209
	s_addc_u32 s11, s7, s3                                     // 000000001778: 820B0307
	s_ashr_i32 s7, s6, 31                                      // 00000000177C: 86079F06
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001780: 84868206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001784: BF870009
	s_add_u32 s0, s0, s6                                       // 000000001788: 80000600
	s_addc_u32 s1, s1, s7                                      // 00000000178C: 82010701
	s_load_b64 s[6:7], s[0:1], null                            // 000000001790: F4040180 F8000000
	s_clause 0x1                                               // 000000001798: BF850001
	s_load_b64 s[12:13], s[10:11], null                        // 00000000179C: F4040305 F8000000
	s_load_b32 s9, s[10:11], 0x8                               // 0000000017A4: F4000245 F8000008
	s_load_b32 s16, s[0:1], 0x8                                // 0000000017AC: F4000400 F8000008
	s_mul_i32 s0, s15, 0x14a                                   // 0000000017B4: 9600FF0F 0000014A
	s_mul_i32 s10, s14, 55                                     // 0000000017BC: 960AB70E
	s_ashr_i32 s1, s0, 31                                      // 0000000017C0: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017C4: BF870499
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017C8: 84808200
	s_add_u32 s4, s4, s0                                       // 0000000017CC: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017D0: 82050105
	s_ashr_i32 s11, s10, 31                                    // 0000000017D4: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017D8: BF870499
	s_lshl_b64 s[0:1], s[10:11], 2                             // 0000000017DC: 8480820A
	s_add_u32 s4, s4, s0                                       // 0000000017E0: 80040004
	s_addc_u32 s5, s5, s1                                      // 0000000017E4: 82050105
	s_waitcnt lgkmcnt(0)                                       // 0000000017E8: BF89FC07
	v_fma_f32 v0, s12, s6, 0                                   // 0000000017EC: D6130000 02000C0C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017F4: BF870091
	v_fmac_f32_e64 v0, s13, s7                                 // 0000000017F8: D52B0000 00000E0D
	v_fmac_f32_e64 v0, s9, s16                                 // 000000001800: D52B0000 00002009
	s_ashr_i32 s9, s8, 31                                      // 000000001808: 86099F08
	v_mov_b32_e32 v1, 0                                        // 00000000180C: 7E020280
	s_lshl_b64 s[0:1], s[8:9], 2                               // 000000001810: 84808208
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001814: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001818: 20000080
	s_add_u32 s0, s4, s0                                       // 00000000181C: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001820: 82010105
	s_add_u32 s0, s0, s2                                       // 000000001824: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001828: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 00000000182C: DC6A0000 00000001
	s_nop 0                                                    // 000000001834: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001838: BFB60003
	s_endpgm                                                   // 00000000183C: BFB00000
