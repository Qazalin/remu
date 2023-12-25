
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_65>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000160C: F4040000 F8000010
	v_mov_b32_e32 v0, 0                                        // 000000001614: 7E000280
	s_mov_b64 s[2:3], 0                                        // 000000001618: BE820180
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s24, s6, s2                                      // 000000001620: 80180206
	s_addc_u32 s25, s7, s3                                     // 000000001624: 82190307
	s_add_u32 s26, s0, s2                                      // 000000001628: 801A0200
	s_addc_u32 s27, s1, s3                                     // 00000000162C: 821B0301
	s_load_b256 s[8:15], s[24:25], null                        // 000000001630: F40C020C F8000000
	s_load_b256 s[16:23], s[26:27], null                       // 000000001638: F40C040D F8000000
	s_add_u32 s2, s2, 52                                       // 000000001640: 8002B402
	s_addc_u32 s3, s3, 0                                       // 000000001644: 82038003
	s_cmpk_eq_i32 s2, 0x104                                    // 000000001648: B1820104
	s_waitcnt lgkmcnt(0)                                       // 00000000164C: BF89FC07
	v_fmac_f32_e64 v0, s8, s16                                 // 000000001650: D52B0000 00002008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001658: BF870091
	v_fmac_f32_e64 v0, s9, s17                                 // 00000000165C: D52B0000 00002209
	v_fmac_f32_e64 v0, s10, s18                                // 000000001664: D52B0000 0000240A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 00000000166C: BF8700B1
	v_fmac_f32_e64 v0, s11, s19                                // 000000001670: D52B0000 0000260B
	s_load_b128 s[8:11], s[24:25], 0x20                        // 000000001678: F408020C F8000020
	s_load_b128 s[16:19], s[26:27], 0x20                       // 000000001680: F408040D F8000020
	v_fmac_f32_e64 v0, s12, s20                                // 000000001688: D52B0000 0000280C
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001690: BF8700B1
	v_fmac_f32_e64 v0, s13, s21                                // 000000001694: D52B0000 00002A0D
	s_load_b32 s12, s[24:25], 0x30                             // 00000000169C: F400030C F8000030
	s_load_b32 s13, s[26:27], 0x30                             // 0000000016A4: F400034D F8000030
	v_fmac_f32_e64 v0, s14, s22                                // 0000000016AC: D52B0000 00002C0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016B4: BF8700A1
	v_fmac_f32_e64 v0, s15, s23                                // 0000000016B8: D52B0000 00002E0F
	s_waitcnt lgkmcnt(0)                                       // 0000000016C0: BF89FC07
	v_fmac_f32_e64 v0, s8, s16                                 // 0000000016C4: D52B0000 00002008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016CC: BF870091
	v_fmac_f32_e64 v0, s9, s17                                 // 0000000016D0: D52B0000 00002209
	v_fmac_f32_e64 v0, s10, s18                                // 0000000016D8: D52B0000 0000240A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016E0: BF870091
	v_fmac_f32_e64 v0, s11, s19                                // 0000000016E4: D52B0000 0000260B
	v_fmac_f32_e64 v0, s12, s13                                // 0000000016EC: D52B0000 00001A0C
	s_cbranch_scc0 65481                                       // 0000000016F4: BFA1FFC9 <r_65+0x1c>
	v_mov_b32_e32 v1, 0                                        // 0000000016F8: 7E020280
	global_store_b32 v1, v0, s[4:5]                            // 0000000016FC: DC6A0000 00040001
	s_nop 0                                                    // 000000001704: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001708: BFB60003
	s_endpgm                                                   // 00000000170C: BFB00000
