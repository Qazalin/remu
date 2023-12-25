
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_6_11_6_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_hi_i32 s8, s13, 0x2aaaaaab                           // 000000001714: 9708FF0D 2AAAAAAB
	s_mul_i32 s2, s15, 0x4d                                    // 00000000171C: 9602FF0F 0000004D
	s_lshr_b32 s9, s8, 31                                      // 000000001724: 85099F08
	s_ashr_i32 s3, s2, 31                                      // 000000001728: 86039F02
	s_add_i32 s9, s8, s9                                       // 00000000172C: 81090908
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_mul_i32 s8, s9, 6                                        // 000000001734: 96088609
	s_mul_i32 s10, s9, 7                                       // 000000001738: 960A8709
	s_sub_i32 s12, s13, s8                                     // 00000000173C: 818C080D
	s_waitcnt lgkmcnt(0)                                       // 000000001740: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001744: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001748: 82070307
	s_ashr_i32 s11, s10, 31                                    // 00000000174C: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[2:3], s[10:11], 2                             // 000000001754: 8482820A
	s_add_u32 s6, s6, s2                                       // 000000001758: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000175C: 82070307
	s_ashr_i32 s13, s12, 31                                    // 000000001760: 860D9F0C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001764: BF8704D9
	s_lshl_b64 s[2:3], s[12:13], 2                             // 000000001768: 8482820C
	s_mul_i32 s12, s14, 0x42                                   // 00000000176C: 960CFF0E 00000042
	s_add_u32 s6, s6, s2                                       // 000000001774: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001778: 82070307
	s_lshl_b32 s10, s14, 1                                     // 00000000177C: 840A810E
	s_ashr_i32 s11, s10, 31                                    // 000000001780: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001784: BF870499
	s_lshl_b64 s[10:11], s[10:11], 2                           // 000000001788: 848A820A
	s_add_u32 s0, s0, s10                                      // 00000000178C: 80000A00
	s_addc_u32 s1, s1, s11                                     // 000000001790: 82010B01
	s_load_b64 s[6:7], s[6:7], null                            // 000000001794: F4040183 F8000000
	s_load_b64 s[0:1], s[0:1], null                            // 00000000179C: F4040000 F8000000
	s_mul_i32 s10, s15, 0x18c                                  // 0000000017A4: 960AFF0F 0000018C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017AC: BF870499
	s_ashr_i32 s11, s10, 31                                    // 0000000017B0: 860B9F0A
	s_lshl_b64 s[10:11], s[10:11], 2                           // 0000000017B4: 848A820A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 0000000017B8: BF8704B9
	s_add_u32 s9, s4, s10                                      // 0000000017BC: 80090A04
	s_addc_u32 s10, s5, s11                                    // 0000000017C0: 820A0B05
	s_ashr_i32 s13, s12, 31                                    // 0000000017C4: 860D9F0C
	s_lshl_b64 s[4:5], s[12:13], 2                             // 0000000017C8: 8484820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)// 0000000017CC: BF8700D9
	s_add_u32 s4, s9, s4                                       // 0000000017D0: 80040409
	s_addc_u32 s5, s10, s5                                     // 0000000017D4: 8205050A
	s_ashr_i32 s9, s8, 31                                      // 0000000017D8: 86099F08
	s_waitcnt lgkmcnt(0)                                       // 0000000017DC: BF89FC07
	v_fma_f32 v0, s6, s0, 0                                    // 0000000017E0: D6130000 02000006
	v_fmac_f32_e64 v0, s7, s1                                  // 0000000017E8: D52B0000 00000207
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000017F0: 84808208
	v_mov_b32_e32 v1, 0                                        // 0000000017F4: 7E020280
	s_add_u32 s0, s4, s0                                       // 0000000017F8: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017FC: 82010105
	v_max_f32_e32 v0, 0, v0                                    // 000000001800: 20000080
	s_add_u32 s0, s0, s2                                       // 000000001804: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001808: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 00000000180C: DC6A0000 00000001
	s_nop 0                                                    // 000000001814: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001818: BFB60003
	s_endpgm                                                   // 00000000181C: BFB00000
