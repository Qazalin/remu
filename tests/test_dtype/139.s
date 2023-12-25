
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_2_2_2n4>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s2, s15, 1                                      // 000000001714: 8402810F
	v_mov_b32_e32 v0, 0                                        // 000000001718: 7E000280
	s_ashr_i32 s3, s2, 31                                      // 00000000171C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001720: BF8704D9
	s_lshl_b64 s[2:3], s[2:3], 1                               // 000000001724: 84828102
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s6, s6, s2                                       // 00000000172C: 80060206
	s_addc_u32 s7, s7, s3                                      // 000000001730: 82070307
	s_lshl_b32 s8, s14, 1                                      // 000000001734: 8408810E
	s_ashr_i32 s9, s8, 31                                      // 000000001738: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_lshl_b64 s[8:9], s[8:9], 1                               // 000000001740: 84888108
	s_add_u32 s0, s0, s8                                       // 000000001744: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001748: 82010901
	s_clause 0x1                                               // 00000000174C: BF850001
	global_load_b32 v1, v0, s[6:7]                             // 000000001750: DC520000 01060000
	global_load_b32 v2, v0, s[0:1]                             // 000000001758: DC520000 02000000
	s_add_u32 s2, s4, s2                                       // 000000001760: 80020204
	s_addc_u32 s3, s5, s3                                      // 000000001764: 82030305
	s_ashr_i32 s15, s14, 31                                    // 000000001768: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000176C: BF870499
	s_lshl_b64 s[0:1], s[14:15], 1                             // 000000001770: 8480810E
	s_add_u32 s0, s2, s0                                       // 000000001774: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001778: 82010103
	s_waitcnt vmcnt(0)                                         // 00000000177C: BF8903F7
	v_pk_mul_lo_u16 v1, v2, v1                                 // 000000001780: CC014001 18020302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001788: BF870091
	v_lshrrev_b32_e32 v2, 16, v1                               // 00000000178C: 32040290
	v_add_nc_u16 v1, v2, v1                                    // 000000001790: D7030001 00020302
	global_store_b16 v0, v1, s[0:1]                            // 000000001798: DC660000 00000100
	s_nop 0                                                    // 0000000017A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017A4: BFB60003
	s_endpgm                                                   // 0000000017A8: BFB00000
