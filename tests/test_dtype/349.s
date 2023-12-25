
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_4n169>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000171C: BF870009
	s_lshl_b64 s[8:9], s[2:3], 3                               // 000000001720: 84888302
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001728: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000172C: 82070907
	s_add_u32 s0, s0, s8                                       // 000000001730: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001734: 82010901
	s_load_b64 s[6:7], s[6:7], null                            // 000000001738: F4040183 F8000000
	s_load_b64 s[0:1], s[0:1], null                            // 000000001740: F4040000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001748: BF89FC07
	s_clz_i32_u32 s8, s7                                       // 00000000174C: BE880A07
	s_xor_b32 s9, s0, s1                                       // 000000001750: 8D090100
	s_cls_i32 s10, s1                                          // 000000001754: BE8A0C01
	s_ashr_i32 s9, s9, 31                                      // 000000001758: 86099F09
	s_add_i32 s10, s10, -1                                     // 00000000175C: 810AC10A
	s_add_i32 s9, s9, 32                                       // 000000001760: 8109A009
	s_min_u32 s8, s8, 32                                       // 000000001764: 8988A008
	s_min_u32 s9, s10, s9                                      // 000000001768: 8989090A
	s_lshl_b64 s[6:7], s[6:7], s8                              // 00000000176C: 84860806
	s_lshl_b64 s[0:1], s[0:1], s9                              // 000000001770: 84800900
	s_min_u32 s6, s6, 1                                        // 000000001774: 89868106
	s_min_u32 s0, s0, 1                                        // 000000001778: 89808100
	s_or_b32 s6, s7, s6                                        // 00000000177C: 8C060607
	s_or_b32 s0, s1, s0                                        // 000000001780: 8C000001
	v_cvt_f32_u32_e32 v0, s6                                   // 000000001784: 7E000C06
	v_cvt_f32_i32_e32 v1, s0                                   // 000000001788: 7E020A00
	s_sub_i32 s0, 32, s8                                       // 00000000178C: 818008A0
	s_sub_i32 s1, 32, s9                                       // 000000001790: 818109A0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 000000001794: BF870112
	v_ldexp_f32 v0, v0, s0                                     // 000000001798: D71C0000 00000100
	v_ldexp_f32 v1, v1, s1                                     // 0000000017A0: D71C0001 00000301
	s_lshl_b64 s[0:1], s[2:3], 1                               // 0000000017A8: 84808102
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000017AC: BF870119
	s_add_u32 s0, s4, s0                                       // 0000000017B0: 80000004
	v_cvt_f16_f32_e32 v0, v0                                   // 0000000017B4: 7E001500
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017B8: BF8700A2
	v_cvt_f16_f32_e32 v1, v1                                   // 0000000017BC: 7E021501
	s_addc_u32 s1, s5, s1                                      // 0000000017C0: 82010105
	v_mul_f16_e32 v0, v0, v1                                   // 0000000017C4: 6A000300 ; Error: VGPR_32_Lo128: unknown register 128
	v_mov_b32_e32 v1, 0                                        // 0000000017C8: 7E020280
	global_store_b16 v1, v0, s[0:1]                            // 0000000017CC: DC660000 00000001
	s_nop 0                                                    // 0000000017D4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D8: BFB60003
	s_endpgm                                                   // 0000000017DC: BFB00000
